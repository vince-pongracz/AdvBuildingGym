"""Reward schedule switching via RLlib on_train_result callback.

Pushes a new reward function subset to all env_runners (training AND
evaluation) at training iteration boundaries.  The active subset is
determined by the RewardScheduleManager's mode and swap index.

Pattern mirrors ``data_schedule_callback.py`` (Approach D1).

``create_reward_switch_on_train_result(...)`` returns an
``on_train_result`` function that can be passed as a keyword argument
to ``config.callbacks(ExistingClass, on_train_result=func)``.

When ``RewardScheduleManager.exploration_bump.enabled`` is set, the
callback also implements the **Dynamic Callback** exploration strategy:
on every reward set change it pushes the entropy term (PPO
``entropy_coeff`` / SAC ``log_alpha``) and optionally the optimiser LRs
up by a configured factor on every Learner, then linearly decays them
back to baseline over ``decay_iterations`` training iterations.  This
re-encourages exploration so the policy can re-test the action space
under the shifted objective.
"""

import logging
import math

from adv_building_gym.config.reward_schedule_manager import (
    ExplorationBumpConfig,
    RewardScheduleManager,
)

logger = logging.getLogger(__name__)


def create_reward_switch_on_train_result_cb(
    reward_manager: RewardScheduleManager,
):
    """Factory that returns an ``on_train_result`` callable.

    Usage::

        config.callbacks(
            SomeCallbackClass,
            on_train_result=create_reward_switch_on_train_result(manager),
        )

    Args:
        reward_manager: Stateful RewardScheduleManager that tracks the
            schedule position and creates reward subsets.
    """

    # Closure state for the dynamic exploration bump:
    #   bump_iter   — training iteration at which the last bump was applied;
    #                 None = no active bump.
    #   baselines_captured — True after we have snapshotted entropy/LR on
    #                 the Learners (deferred until first on_train_result so
    #                 the LearnerGroup is fully built).
    state: dict = {
        "bump_iter": None, 
        "baselines_captured": False
    }

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        logger.info("RewardScheduleManager iteration: %d", iteration)

        bump_cfg = reward_manager.exploration_bump
        if bump_cfg.enabled and not state["baselines_captured"]:
            _capture_baselines(algorithm)
            state["baselines_captured"] = True

        # Linear decay back toward baseline. Runs every iteration while a
        # bump is active so the entropy / LR envelope ramps down smoothly
        # before the next swap.
        if bump_cfg.enabled and state["bump_iter"] is not None:
            elapsed = iteration - state["bump_iter"]
            if elapsed >= bump_cfg.decay_iterations:
                _apply_exploration_level(algorithm, bump_cfg, frac=0.0)
                state["bump_iter"] = None
                logger.info(
                    "Exploration bump decayed back to baseline (iter=%d).",
                    iteration,
                )
            else:
                frac = 1.0 - (elapsed / bump_cfg.decay_iterations)
                _apply_exploration_level(algorithm, bump_cfg, frac=frac)

        if iteration % reward_manager.swap_every_n_iterations != 0:
            return

        changed = reward_manager.advance()
        if not changed:
            return

        _push_rewards_to_runners(algorithm, reward_manager)

        if bump_cfg.enabled:
            _apply_exploration_level(algorithm, bump_cfg, frac=1.0)
            state["bump_iter"] = iteration
            logger.info(
                "Exploration bump applied on reward swap (iter=%d): "
                "entropy_coeff target=%.4g, sac_alpha target=%.4g, lr_mult=%.3g, "
                "decay over %d iters.",
                iteration, bump_cfg.ppo_entropy_coeff, bump_cfg.sac_alpha,
                bump_cfg.lr_multiplier, bump_cfg.decay_iterations,
            )

    return on_train_result


def _push_rewards_to_runners(
    algorithm,
    reward_manager: RewardScheduleManager,
) -> None:
    """Create fresh reward instances and set them on all live envs.

    Both training and evaluation env_runners receive the same active
    reward set so that eval metrics reflect the current training objective.
    """

    def apply(env_runner) -> None:
        # env_runner.env is wrapped:
        #   DictInfoToList -> SyncVectorEnv -> [TimeLimit -> ... -> AdvBuildingGym]
        # The local (driver) env_runner may have env=None in the new API stack.
        vec_env = getattr(env_runner, "env", None)
        if vec_env is None:
            return
        # Unwrap DictInfoToList to reach SyncVectorEnv
        sync_vec = getattr(vec_env, "env", vec_env)
        for sub_env in getattr(sync_vec, "envs", []):
            unwrapped = sub_env.unwrapped
            if hasattr(unwrapped, "set_reward_funcs"):
                # Each sub-env gets its own fresh reward instances
                rewards = reward_manager.create_active_rewards()
                unwrapped.set_reward_funcs(rewards)

    algorithm.env_runner_group.foreach_env_runner(
        apply, local_env_runner=True, timeout_seconds=None,
    )
    # Eval runners get the SAME reward set as training runners.
    # Eval does not influence the policy gradient — it is purely for
    # monitoring — but must evaluate the same objective for metrics to
    # be meaningful.
    if algorithm.eval_env_runner_group is not None:
        algorithm.eval_env_runner_group.foreach_env_runner(
            apply, local_env_runner=True, timeout_seconds=None,
        )

    logger.info(
        "Reward schedule: pushed active rewards [ %s ] to all env_runners",
        reward_manager.get_active_reward_names(),
    )


# ----------------------------------------------------------------------
# Dynamic exploration bump (event-driven)
# ----------------------------------------------------------------------


def _capture_baselines(algorithm) -> None:
    """Snapshot per-learner baseline entropy_coeff and LRs.

    Stored as ``learner._reward_swap_baselines`` (a small dict) so that
    later decay calls can interpolate without needing the original
    AlgorithmConfig.  Run once, on the first on_train_result.
    """

    def capture(learner) -> None:
        baselines: dict = {}
        # PPO: entropy_coeff lives on AlgorithmConfig and is read each loss step.
        cfg = getattr(learner, "config", None)
        if cfg is not None and hasattr(cfg, "entropy_coeff"):
            try:
                baselines["entropy_coeff"] = float(cfg.entropy_coeff)
            except (TypeError, ValueError):
                # entropy_coeff_schedule (list-of-pairs) — leave alone.
                pass

        # SAC: log_alpha is an nn.Parameter on the RLModule; only present
        # when entropy is auto-tuned (default). Capture per module_id so
        # multi-module setups still work.
        log_alpha_baselines: dict[str, float] = {}
        module_dict = getattr(learner, "module", None)
        if module_dict is not None:
            try:
                module_ids = list(module_dict.keys())
            except AttributeError:
                module_ids = []
            for mid in module_ids:
                module = module_dict[mid]
                la = getattr(module, "log_alpha", None)
                if la is not None and hasattr(la, "item"):
                    log_alpha_baselines[mid] = float(la.item())
        if log_alpha_baselines:
            baselines["log_alpha"] = log_alpha_baselines

        # Optimiser LRs (TorchLearner._named_optimizers).
        lr_baselines: dict[str, list[float]] = {}
        named_opts = getattr(learner, "_named_optimizers", None) or {}
        for name, opt in named_opts.items():
            lr_baselines[name] = [float(g["lr"]) for g in opt.param_groups]
        baselines["lrs"] = lr_baselines

        learner._reward_swap_baselines = baselines  # type: ignore[attr-defined]
        logger.info(
            "Captured exploration baselines on learner: "
            "entropy_coeff=%s, log_alpha=%s, optimisers=%s",
            baselines.get("entropy_coeff"),
            log_alpha_baselines or None,
            list(lr_baselines.keys()) or None,
        )

    algorithm.learner_group.foreach_learner(capture)


def _apply_exploration_level(
    algorithm,
    bump_cfg: ExplorationBumpConfig,
    *,
    frac: float,
) -> None:
    """Set entropy / log_alpha / LRs to baseline + frac * (boost - baseline).

    ``frac=1.0`` → fully boosted (just after a swap).
    ``frac=0.0`` → restored to captured baseline.
    """
    frac = max(0.0, min(1.0, float(frac)))

    def apply(learner) -> None:
        baselines = getattr(learner, "_reward_swap_baselines", None)
        if baselines is None:
            return

        # PPO entropy_coeff. AlgorithmConfig may be frozen (RLlib sets
        # _is_frozen=True after .build()) — bypass __setattr__ via object.
        if "entropy_coeff" in baselines:
            base = baselines["entropy_coeff"]
            target = base + frac * (bump_cfg.ppo_entropy_coeff - bump_cfg.ppo_entropy_baseline)
            cfg = getattr(learner, "config", None)
            if cfg is not None:
                try:
                    object.__setattr__(cfg, "entropy_coeff", float(target))
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning("Failed to set entropy_coeff: %s", exc)

        # SAC log_alpha — set in-place on the parameter tensor.
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
                    # Mutate parameter data (no autograd graph; this is a
                    # hyperparameter override, not a training step).
                    la.data.fill_(float(interp))

        # Optimiser LRs.
        lr_baselines = baselines.get("lrs", {})
        if lr_baselines and bump_cfg.lr_multiplier != 1.0:
            named_opts = getattr(learner, "_named_optimizers", None) or {}
            # multiplier interpolates between 1.0 (baseline) and lr_multiplier.
            mult = 1.0 + frac * (bump_cfg.lr_multiplier - 1.0)
            for name, opt in named_opts.items():
                base_lrs = lr_baselines.get(name)
                if base_lrs is None:
                    continue
                for g, base_lr in zip(opt.param_groups, base_lrs):
                    g["lr"] = float(base_lr * mult)

    algorithm.learner_group.foreach_learner(apply)