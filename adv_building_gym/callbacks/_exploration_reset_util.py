"""Shared helpers for the event-driven exploration reset.

Used by reward / infra / statesource swap callbacks.  The owning
callback decides whether to fire (via ``ExplorationResetConfig.fires_on``)
and only then drives ``capture_baselines`` / ``apply_exploration_level``.
"""

from __future__ import annotations

import logging
import math

from adv_building_gym.config.exploration_reset import ExplorationResetConfig

logger = logging.getLogger(__name__)


def capture_baselines(algorithm) -> None:
    """Snapshot per-learner baseline entropy_coeff, log_alpha, and LRs.

    Ratchet semantics: if a baseline already exists, each scalar is updated
    only when the current learned value is *lower* (more focused) than the
    stored one. This preserves the policy's learned focus across swap events
    — bumped exploration always decays back to the tightest baseline seen so
    far, never to a stale, looser early-training snapshot.
    """

    def capture(learner) -> None:
        existing = getattr(learner, "_exploration_reset_baselines", None) or {}
        baselines: dict = {}

        cfg = getattr(learner, "config", None)
        if cfg is not None and hasattr(cfg, "entropy_coeff"):
            try:
                current = float(cfg.entropy_coeff)
            except (TypeError, ValueError):
                current = None
            if current is not None:
                prev = existing.get("entropy_coeff")
                baselines["entropy_coeff"] = current if prev is None else min(prev, current)

        log_alpha_baselines: dict[str, float] = dict(existing.get("log_alpha", {}))
        module_dict = getattr(learner, "module", None)
        if module_dict is not None:
            try:
                module_ids = list(module_dict.keys())
            except AttributeError:
                module_ids = []
            for mid in module_ids:
                module = module_dict[mid]
                log_alpha = getattr(module, "log_alpha", None)
                if log_alpha is not None and hasattr(log_alpha, "item"):
                    current = float(log_alpha.item())
                    prev = log_alpha_baselines.get(mid)
                    log_alpha_baselines[mid] = current if prev is None else min(prev, current)
        if log_alpha_baselines:
            baselines["log_alpha"] = log_alpha_baselines

        lr_baselines: dict[str, list[float]] = {
            name: list(vals) for name, vals in existing.get("lrs", {}).items()
        }
        named_opts = getattr(learner, "_named_optimizers", None) or {}
        for name, opt in named_opts.items():
            current_lrs = [float(g["lr"]) for g in opt.param_groups]
            prev_lrs = lr_baselines.get(name)
            if prev_lrs is None:
                lr_baselines[name] = current_lrs
            else:
                lr_baselines[name] = [
                    min(p, c) for p, c in zip(prev_lrs, current_lrs)
                ]
        baselines["lrs"] = lr_baselines

        learner._exploration_reset_baselines = baselines  # type: ignore[attr-defined]
        logger.info(
            "Captured exploration baselines on learner (ratcheted): "
            "entropy_coeff=%s, log_alpha=%s, optimisers=%s",
            baselines.get("entropy_coeff"), log_alpha_baselines or None,
            list(lr_baselines.keys()) or None,
        )

    algorithm.learner_group.foreach_learner(capture)


def apply_exploration_level(
    algorithm,
    bump_cfg: ExplorationResetConfig,
    *,
    frac: float,
) -> None:
    """Set entropy / log_alpha / LR to baseline + frac * (boost - baseline)."""
    frac = max(0.0, min(1.0, float(frac)))

    def apply(learner) -> None:
        baselines = getattr(learner, "_exploration_reset_baselines", None)
        if baselines is None:
            return

        if "entropy_coeff" in baselines:
            base = baselines["entropy_coeff"]
            target = base + frac * (bump_cfg.ppo_entropy_coeff - bump_cfg.ppo_entropy_baseline)
            cfg = getattr(learner, "config", None)
            if cfg is not None:
                try:
                    object.__setattr__(cfg, "entropy_coeff", float(target))
                except Exception as exc:
                    logger.warning("Failed to set entropy_coeff: %s", exc)

        log_alpha_baselines = baselines.get("log_alpha")
        if log_alpha_baselines:
            target_log = math.log(bump_cfg.sac_alpha)
            module_dict = getattr(learner, "module", None)
            if module_dict is not None:
                for mid, base_log in log_alpha_baselines.items():
                    interp = base_log + frac * (target_log - base_log)
                    try:
                        module = module_dict[mid]
                    except KeyError:
                        continue
                    la = getattr(module, "log_alpha", None)
                    if la is None:
                        continue
                    la.data.fill_(float(interp))

        lr_baselines = baselines.get("lrs", {})
        if lr_baselines and bump_cfg.lr_multiplier != 1.0:
            named_opts = getattr(learner, "_named_optimizers", None) or {}
            mult = 1.0 + frac * (bump_cfg.lr_multiplier - 1.0)
            for name, opt in named_opts.items():
                base_lrs = lr_baselines.get(name)
                if base_lrs is None:
                    continue
                for g, base_lr in zip(opt.param_groups, base_lrs):
                    g["lr"] = float(base_lr * mult)

    algorithm.learner_group.foreach_learner(apply)


def make_decay_loop(exploration_reset: ExplorationResetConfig, event: str):
    """Return a tuple (state, on_train_result_handler) implementing the decay envelope.

    The decay handler should be called every iteration; it decays a
    previously-applied bump back to baseline. Baselines are (re)captured
    inside ``fire_bump`` with ratchet semantics — each swap can only
    *tighten* the baseline towards a more focused learned state.

    Returns (state_dict, maybe_decay, fire_bump). ``state_dict`` keeps
    bump_iter.  ``fire_bump(algorithm, iteration)`` should be called by the
    owning callback right after a successful swap.
    """
    state: dict = {"bump_iter": None}
    fires_here = exploration_reset.fires_on(event)

    def maybe_decay(algorithm, iteration: int) -> None:
        if not fires_here or state["bump_iter"] is None:
            return
        elapsed = iteration - state["bump_iter"]
        if elapsed >= exploration_reset.decay_iterations:
            apply_exploration_level(algorithm, exploration_reset, frac=0.0)
            state["bump_iter"] = None
            logger.info("Exploration bump decayed back to baseline (iter=%d).", iteration)
        else:
            frac = 1.0 - (elapsed / exploration_reset.decay_iterations)
            apply_exploration_level(algorithm, exploration_reset, frac=frac)

    def fire_bump(algorithm, iteration: int) -> None:
        if not fires_here:
            return
        # Ratchet baselines towards the current (possibly more-focused) learner state
        # before applying the bump. capture_baselines() only lowers existing entries.
        capture_baselines(algorithm)

        def _snapshot(learner):
            b = getattr(learner, "_exploration_reset_baselines", None) or {}
            ent = b.get("entropy_coeff")
            log_alphas = b.get("log_alpha") or {}
            alpha = math.exp(next(iter(log_alphas.values()))) if log_alphas else None
            lr = None
            for vals in (b.get("lrs") or {}).values():
                if vals:
                    lr = vals[0]
                    break
            return ent, alpha, lr

        snaps = algorithm.learner_group.foreach_learner(_snapshot) or []
        ent_was, alpha_was, lr_was = snaps[0] if snaps else (None, None, None)

        apply_exploration_level(algorithm, exploration_reset, frac=1.0)
        state["bump_iter"] = iteration

        ent_now = exploration_reset.ppo_entropy_coeff
        alpha_now = exploration_reset.sac_alpha
        lr_now = lr_was * exploration_reset.lr_multiplier if lr_was is not None else None

        def _g(v):
            return "n/a".rjust(10) if v is None else f"{v:10.4g}"

        logger.info(
            "Exploration bump applied (event=%s, iter=%d, decay over %d iters):\n"
            "  was:    entropy_coeff=%s  sac_alpha=%s  lr=%s\n"
            "  actual: entropy_coeff=%s  sac_alpha=%s  lr=%s  (lr_mult=%.3g)",
            event, iteration, exploration_reset.decay_iterations,
            _g(ent_was), _g(alpha_was), _g(lr_was),
            _g(ent_now), _g(alpha_now), _g(lr_now), exploration_reset.lr_multiplier,
        )

    return state, maybe_decay, fire_bump
