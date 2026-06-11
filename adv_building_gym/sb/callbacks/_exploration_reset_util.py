"""Shared helpers for the SB3 event-driven exploration reset.

SB3 port of :mod:`adv_building_gym.ray.callbacks._exploration_reset_util`.

Used by the reward / infra / statesource swap callbacks. The owning
callback decides whether to fire (via ``ExplorationResetConfig.fires_on``)
and only then drives ``capture_baselines`` / ``apply_exploration_level``.

Differences vs. the RLlib counterpart
-------------------------------------

RLlib's helper iterates over ``algorithm.learner_group.foreach_learner``
(possibly many remote actors). SB3 has a single in-process model, so
``model`` is the only target.

PPO    — bumps ``model.ent_coef`` (plain float coefficient).
SAC    — bumps ``model.log_ent_coef`` (an ``nn.Parameter``;
         ``ent_coef = exp(log_ent_coef)``). When SAC was built with
         ``ent_coef='auto'``, this parameter is updated *during training*
         by ``model.ent_coef_optimizer``; the bump still raises the live
         value and the auto-tuner gradually pulls it back towards target
         — exactly the dynamic we want.
LR mult — multiplies the LR of every param-group in every optimizer
         attached to ``model.policy`` (actor / critic / log_alpha / etc.).

Decay envelope mirrors the Ray version: on swap, capture the
*tightest* baselines seen so far (ratchet), apply the boost, then
linearly decay back towards baseline over ``decay_iterations``
callback ticks. SB3 has no algorithm-iteration boundary so each
``_on_step`` counts as one tick — this is denser than RLlib but
the linear envelope keeps the integral matched.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import torch as th

from adv_building_gym.config.training.exploration_reset import ExplorationResetConfig

logger = logging.getLogger(__name__)


# Custom attribute we attach to the SB3 model to stash baselines + decay state
# without polluting the algorithm class.
_BASELINES_ATTR = "_sb_exploration_reset_baselines"
_DECAY_STATE_ATTR = "_sb_exploration_reset_decay_state"


def _iter_optimizers(model) -> Iterable[tuple[str, th.optim.Optimizer]]:
    """Yield (name, optimizer) for each optimizer on the SB3 model
    (policy / actor / critic / ent_coef as present)."""
    policy = getattr(model, "policy", None)
    if policy is None:
        return
    seen: set[int] = set()

    def _emit(name: str, holder):
        opt = getattr(holder, "optimizer", None)
        if opt is None or id(opt) in seen:
            return
        seen.add(id(opt))
        yield (name, opt)

    yield from _emit("policy", policy)
    actor = getattr(policy, "actor", None)
    if actor is not None:
        yield from _emit("actor", actor)
    critic = getattr(policy, "critic", None)
    if critic is not None:
        yield from _emit("critic", critic)
    ent_coef_opt = getattr(model, "ent_coef_optimizer", None)
    if ent_coef_opt is not None and id(ent_coef_opt) not in seen:
        seen.add(id(ent_coef_opt))
        yield ("ent_coef", ent_coef_opt)


def capture_baselines(model) -> None:
    """Snapshot baseline entropy_coeff / log_alpha / per-optimiser LRs.

    Ratchet: existing baselines are only lowered, so bumps decay back to the tightest seen.
    """
    existing: dict = getattr(model, _BASELINES_ATTR, None) or {}
    baselines: dict = {}

    # PPO ent_coef is a float; SAC's resolves to log_ent_coef, routed through log_alpha below
    ent_coef = getattr(model, "ent_coef", None)
    if isinstance(ent_coef, (int, float)):
        current = float(ent_coef)
        prev = existing.get("entropy_coeff")
        baselines["entropy_coeff"] = current if prev is None else min(prev, current)

    # SAC log_alpha tensor.
    log_ent_coef = getattr(model, "log_ent_coef", None)
    if isinstance(log_ent_coef, th.Tensor):
        current = float(log_ent_coef.detach().cpu().item())
        prev = existing.get("log_alpha")
        baselines["log_alpha"] = current if prev is None else min(prev, current)

    # Per-optimiser LRs, indexed by optimiser name -> list of param_group LRs.
    lr_baselines: dict[str, list[float]] = {
        n: list(v) for n, v in (existing.get("lrs") or {}).items()
    }
    for name, opt in _iter_optimizers(model):
        current_lrs = [float(g["lr"]) for g in opt.param_groups]
        prev_lrs = lr_baselines.get(name)
        lr_baselines[name] = (
            current_lrs
            if prev_lrs is None
            else [min(p, c) for p, c in zip(prev_lrs, current_lrs)]
        )
    if lr_baselines:
        baselines["lrs"] = lr_baselines

    setattr(model, _BASELINES_ATTR, baselines)
    logger.info(
        "SB exploration reset: captured baselines (ratcheted) "
        "entropy_coeff=%s, log_alpha=%s, optimisers=%s",
        baselines.get("entropy_coeff"),
        baselines.get("log_alpha"),
        list(lr_baselines.keys()) or None,
    )


def apply_exploration_level(
    model,
    bump_cfg: ExplorationResetConfig,
    *,
    frac: float,
) -> None:
    """Set entropy / log_alpha / LR to baseline + frac*(boost - baseline); frac 1=boost, 0=baseline."""
    frac = max(0.0, min(1.0, float(frac)))
    baselines: dict = getattr(model, _BASELINES_ATTR, None) or {}

    if "entropy_coeff" in baselines and hasattr(model, "ent_coef"):
        base = baselines["entropy_coeff"]
        target = base + frac * (bump_cfg.ppo_entropy_coeff - bump_cfg.ppo_entropy_baseline)
        try:
            model.ent_coef = float(target)
        except (AttributeError, TypeError) as exc:
            logger.warning("SB exploration reset: failed to set ent_coef: %s", exc)

    log_ent_coef = getattr(model, "log_ent_coef", None)
    if "log_alpha" in baselines and isinstance(log_ent_coef, th.Tensor):
        base_log = baselines["log_alpha"]
        target_log = math.log(bump_cfg.sac_alpha)
        interp = base_log + frac * (target_log - base_log)
        # in-place edit of the nn.Parameter so the next training step picks it up
        with th.no_grad():
            log_ent_coef.data.fill_(float(interp))

    if bump_cfg.lr_multiplier != 1.0 and "lrs" in baselines:
        mult = 1.0 + frac * (bump_cfg.lr_multiplier - 1.0)
        for name, opt in _iter_optimizers(model):
            base_lrs = baselines["lrs"].get(name)
            if base_lrs is None:
                continue
            for group, base_lr in zip(opt.param_groups, base_lrs):
                group["lr"] = float(base_lr * mult)


def make_decay_loop(exploration_reset: ExplorationResetConfig, *, event: str):
    """Return ``(maybe_decay, fire_bump)``.

    ``maybe_decay(model)`` runs every ``_on_step`` and linearly decays an in-flight bump.
    ``fire_bump(model)`` runs after a swap (when ``fires_on(event)``). State lives on the model
    so each callback has its own decay clock. Decay length: ``decay_iterations`` ticks.
    """

    fires_here = exploration_reset.fires_on(event)

    def _state(model) -> dict:
        state = getattr(model, _DECAY_STATE_ATTR, None)
        if state is None:
            state = {}
            setattr(model, _DECAY_STATE_ATTR, state)
        return state

    def maybe_decay(model) -> None:
        if not fires_here:
            return
        state = _state(model)
        bump_step = state.get(event)
        if bump_step is None:
            return
        elapsed = int(getattr(model, "num_timesteps", 0)) - int(bump_step)
        if elapsed >= exploration_reset.decay_iterations:
            apply_exploration_level(model, exploration_reset, frac=0.0)
            state.pop(event, None)
            logger.info(
                "SB exploration reset (%s): bump decayed to baseline "
                "(elapsed=%d steps).",
                event, elapsed,
            )
            return
        frac = 1.0 - (elapsed / exploration_reset.decay_iterations)
        apply_exploration_level(model, exploration_reset, frac=frac)

    def fire_bump(model) -> None:
        if not fires_here:
            return
        # ratchet baselines to the current state before the bump
        capture_baselines(model)
        apply_exploration_level(model, exploration_reset, frac=1.0)
        _state(model)[event] = int(getattr(model, "num_timesteps", 0))

        # Snapshot post-bump values for the log line.
        ent_now = getattr(model, "ent_coef", None)
        log_alpha = getattr(model, "log_ent_coef", None)
        alpha_now = (
            float(th.exp(log_alpha.detach()).cpu().item())
            if isinstance(log_alpha, th.Tensor) else None
        )
        logger.info(
            "SB exploration reset (%s) fired at step=%d: "
            "ent_coef -> %s, sac_alpha -> %s, lr_mult=%.3g, decay over %d steps.",
            event, int(getattr(model, "num_timesteps", 0)),
            f"{float(ent_now):.4g}" if isinstance(ent_now, (int, float)) else "n/a",
            f"{alpha_now:.4g}" if alpha_now is not None else "n/a",
            exploration_reset.lr_multiplier, exploration_reset.decay_iterations,
        )

    return maybe_decay, fire_bump
