"""
Environment creator factory function for Ray RLlib/Tune.

This module provides the factory function used by Ray Tune to create
AdvBuildingGym environment instances with the configured settings.
"""

import itertools
import logging
import os
import gymnasium

from adv_building_gym.components.rewards import SumRewardAggregator

from adv_building_gym.core.env import AdvBuildingGym
from adv_building_gym.core.forecast_wrapper import ForecastWrapper
from adv_building_gym.core.history_wrapper import HistoryWrapper
from adv_building_gym.ray.ma_env import MultiAgentAdvBuildingGym
from adv_building_gym.core.wrappers import FlattenAction, wrap_action_space

logger = logging.getLogger(__name__)

# Fallback counter for the multi-agent creator when no EnvContext metadata
# is available (local / eval / test path).
_env_instance_counter = itertools.count()


def merge_env_context(base: dict, cfg):
    """Merge the static creator config with RLlib's EnvContext WITHOUT
    dropping its ``worker_index`` / ``vector_index``.

    A plain ``{**base, **cfg}`` returns a bare ``dict`` and silently loses
    those attributes (they live on EnvContext as attributes, not dict items),
    which is why the creators previously fell back to per-process counters /
    ``os.getpid()``.  Re-wrapping into an EnvContext preserves the metadata.
    """
    merged = {**base, **cfg}
    # Deferred import so non-Ray callers (SB driver, tests) don't pull in ray.
    # TODO VP 2026.05.31.: But what uses the SB driver?
    from ray.rllib.env.env_context import EnvContext
    if isinstance(cfg, EnvContext):
        return EnvContext(
            merged,
            worker_index=cfg.worker_index,
            vector_index=cfg.vector_index,
            num_workers=cfg.num_workers,
            remote=cfg.remote,
            recreated_worker=cfg.recreated_worker,
        )
    return merged  # local / eval / test path: plain dict, no metadata


def adv_building_env_creator(config: dict) -> gymnasium.Env:
    """Factory function for Ray Tune to create AdvBuildingGym instances.

    This function is registered with Ray Tune and called whenever a new
    environment instance is needed (e.g., for env runners, evaluation).

    Uses factory methods to create FRESH component instances for each
    environment.  This ensures parallel env_runners don't share mutable
    state (iteration counters, internal buffers).

    Args:
        config: Configuration dict passed by Ray Tune. Required keys:
            - ``env_config``: EnvConfig instance carrying the YAML-loaded
              component specs. Each call deserialises fresh component
              instances from those specs.
            - ``reward_schedule_manager``: RewardScheduleManager instance.
              Rewards are created from the manager's active subset
              (mode=OFF returns all rewards).
          Optional keys:
            - ``data_combinator``: Pre-built DataCombinator instance.
            - ``log_full_info``: When True, enables deep-copy of named
              state into info["state"] each step (evaluation only).

    Returns:
        Wrapped AdvBuildingGym with flat Box(-1, 1) action space.
    """
    env_config = config.get("env_config")
    if env_config is None:
        raise ValueError(
            "adv_building_env_creator: 'env_config' missing from creator config. "
            "The training driver must register the env with the active EnvConfig "
            "(e.g. env_creator_config={'env_config': active_config, ...})."
        )

    # Create fresh instances for this environment from the YAML-loaded specs.
    # Each env gets its own infras/statesources/rewards with independent state.
    infras = env_config.create_infras()
    statesources = env_config.create_statesources()

    # Rewards always come from the RewardScheduleManager (mode=OFF returns all).
    reward_manager = config["reward_schedule_manager"]
    rewards = reward_manager.create_active_rewards()

    # Real RLlib indices, preserved through merge_env_context() at the
    # registration site. worker_index: 0 = local runner, 1..N = remote
    # runners; vector_index = sub-env slot within the worker.
    worker_index = getattr(config, "worker_index", 0)
    vector_index = getattr(config, "vector_index", 0)
    instance_id = f"AdvBuildingGym_w{worker_index}_v{vector_index}"

    # DEBUG: confirm the EnvContext metadata actually reaches the creator.
    # "MISSING" means the indices were dropped before this point (cfg type=dict).
    logger.info(
        "env_creator: worker_index=%s vector_index=%s recreated=%s num_workers=%s "
        "(cfg type=%s) → instance_id=%s",
        getattr(config, "worker_index", "MISSING"), getattr(config, "vector_index", "MISSING"),
        getattr(config, "recreated_worker", "?"), getattr(config, "num_workers", "?"),
        type(config).__name__, instance_id,
    )

    env = AdvBuildingGym(
        infras=infras,
        statesources=statesources,
        rewards=rewards,
        env_config=env_config,
        data_combinator=config.get("data_combinator"),
        reward_aggregator=SumRewardAggregator(),
        instance_id=instance_id,
    )
    seed = config.get("seed", 21) + worker_index + 1000 * vector_index  # Derive a unique seed per env instance.
    env.reset(seed=seed)
    logger.info("Env: instance_id=%s seed=%s", instance_id, seed)
    
    # Seed action space if the method exists
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    # Seed observation space if the method exists
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    logger.info("env_creator: instance_id=%s (config type=%s)",
            instance_id, type(config).__name__)

    # Set by Ray's evaluation env_config — only eval EnvRunners pass these.
    if config.get("log_full_info", False):
        env.log_full_info = True
    if config.get("eval_mode", False):
        env.eval_mode = True

    if env_config.hst_env_wrapper_enabled:
        env = HistoryWrapper(
            env,
            tracked_keys=env_config.hst_env_wrapper_tracked_keys,
            offsets=env_config.hst_env_wrapper_offsets,
        )
        logger.info(
            "env_creator: HistoryWrapper enabled (tracked_keys=%s, offsets=%s)",
            list(env_config.hst_env_wrapper_tracked_keys),
            list(env_config.hst_env_wrapper_offsets),
        )

    if env_config.forecast_env_wrapper_enabled:
        env = ForecastWrapper(env, forecast_steps=env_config.forecast_env_wrapper_steps)
        logger.info(
            "env_creator: ForecastWrapper enabled (steps=%s)",
            list(env_config.forecast_env_wrapper_steps),
        )

    return wrap_action_space(env)


def adv_building_ma_env_creator(config: dict):
    """Factory for the per-actuator multi-agent variant.

    Mirrors :func:`adv_building_env_creator` but builds a
    :class:`MultiAgentAdvBuildingGym` and skips the flat-Box action
    wrappers — per-agent rescale lives inside the MA wrapper.

    Optional config key:
        ``reward_partition``: dict[agent_id → list[reward_name]] mapping
            distributing reward breakdown components per agent. ``None``
            (default) routes the global aggregated reward to every agent
            (cooperative MARL).
    """
    env_config = config.get("env_config")
    if env_config is None:
        raise ValueError(
            "adv_building_ma_env_creator: 'env_config' missing from creator config."
        )

    infras = env_config.create_infras()
    statesources = env_config.create_statesources()
    reward_manager = config["reward_schedule_manager"]
    rewards = reward_manager.create_active_rewards()

    # DEBUG: confirm the EnvContext metadata actually reaches the creator.
    # "MISSING" here means the indices were dropped before this point and
    # the os.getpid() fallback below will fire.
    logger.info(
        "ma_env_creator: worker_index=%s vector_index=%s recreated=%s (cfg type=%s)",
        getattr(config, "worker_index", "MISSING"),
        getattr(config, "vector_index", "MISSING"),
        getattr(config, "recreated_worker", "?"),
        type(config).__name__,
    )

    worker_index = getattr(config, "worker_index", config.get("worker_index", None))
    vector_index = getattr(config, "vector_index", config.get("vector_index", None))
    if worker_index is None or vector_index is None:
        logger.warning("ma_env_creator: no EnvContext indices — falling back to os.getpid().")
        worker_index = os.getpid()
        vector_index = next(_env_instance_counter)
    instance_id = f"AdvBuildingGymMA_w{worker_index}_v{vector_index}"

    env = MultiAgentAdvBuildingGym(
        infras=infras,
        statesources=statesources,
        rewards=rewards,
        env_config=env_config,
        data_combinator=config.get("data_combinator"),
        instance_id=instance_id,
        reward_partition=config.get("reward_partition"),
    )

    seed = config.get("seed", 21) + worker_index + 1000 * vector_index  # Unique per-env construction seed.
    env.reset(seed=seed)
    logger.info("MA Env: instance_id=%s seed=%s", instance_id, seed)

    if config.get("log_full_info", False):
        env.log_full_info = True

    if env_config.hst_env_wrapper_enabled:
        # HistoryWrapper targets the single-agent Dict obs space; the
        # multi-agent variant has a per-agent space and is not supported.
        raise NotImplementedError("HistoryWrapper is not supported in the multi-agent env creator.")

    if env_config.forecast_env_wrapper_enabled:
        raise NotImplementedError("ForecastWrapper is not supported in the multi-agent env creator.")

    logger.info("ma_env_creator: instance_id=%s agents=%s",
                instance_id, env.possible_agents)
    return env
