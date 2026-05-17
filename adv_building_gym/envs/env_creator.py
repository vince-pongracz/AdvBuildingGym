"""
Environment creator factory function for Ray RLlib/Tune.

This module provides the factory function used by Ray Tune to create
AdvBuildingGym environment instances with the configured settings.
"""

import itertools
import logging
import os
import gymnasium

from gymnasium.wrappers import RescaleAction

from adv_building_gym.rewards import SumRewardAggregator

from .building_adv import AdvBuildingGym
from .forecast_wrapper import ForecastWrapper
from .history_wrapper import HistoryWrapper
from .multi_agent_building import MultiAgentAdvBuildingGym
from .wrappers import FlattenAction

logger = logging.getLogger(__name__)

# Per-process counter for the vector_index portion of instance_id. RLlib's new
# API stack passes a plain dict (not EnvContext) to env_creator, so worker_index
# and vector_index are unavailable. PID is unique per Ray remote worker; this
# counter disambiguates multiple envs created within the same process (when
# num_envs_per_env_runner > 1).
_env_instance_counter = itertools.count()

def wrap_action_space(env: gymnasium.Env) -> gymnasium.Env:
    """Apply FlattenAction + RescaleAction wrapper chain.

    The resulting env exposes a flat Box(-1, 1) action space to the policy
    while the inner AdvBuildingGym receives named Dict actions with real
    component bounds.
    """
    env = FlattenAction(env)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    return env


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

    # New-API-stack env_runners pass a plain dict here (no EnvContext), so
    # worker_index/vector_index aren't available. Use PID + a process-local
    # counter to give each env instance a globally unique caller_id in the
    # RngService registry — making per-env seeds reproducible regardless of
    # RPC arrival order. Prefer EnvContext attrs when present (legacy stack).
    worker_index = getattr(config, "worker_index", config.get("worker_index", None))
    vector_index = getattr(config, "vector_index", config.get("vector_index", None))
    if worker_index is None or vector_index is None:
        worker_index = os.getpid()
        vector_index = next(_env_instance_counter)
    instance_id = f"AdvBuildingGym_w{worker_index}_v{vector_index}"

    env = AdvBuildingGym(
        infras=infras,
        statesources=statesources,
        rewards=rewards,
        env_config=env_config,
        data_combinator=config.get("data_combinator"),
        reward_aggregator=SumRewardAggregator(),
        instance_id=instance_id,
    )
    
    logger.info("env_creator: instance_id=%s (config type=%s)",
            instance_id, type(config).__name__)

    # Set by Ray's evaluation env_config — only eval EnvRunners pass this.
    if config.get("log_full_info", False):
        env.log_full_info = True

    if env_config.hst_env_wrapper_enabled:
        env = HistoryWrapper(env, hst_len=env_config.hst_env_wrapper_hst_len)
        logger.info(
            "env_creator: HistoryWrapper enabled (hst_len=%d)",
            env_config.hst_env_wrapper_hst_len,
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

    worker_index = getattr(config, "worker_index", config.get("worker_index", None))
    vector_index = getattr(config, "vector_index", config.get("vector_index", None))
    if worker_index is None or vector_index is None:
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

    if config.get("log_full_info", False):
        env.log_full_info = True

    if env_config.hst_env_wrapper_enabled:
        # HistoryWrapper targets the single-agent Dict obs space; the
        # multi-agent variant has a per-agent space and is not supported.
        raise NotImplementedError(
            "HistoryWrapper is not supported in the multi-agent env creator."
        )

    if env_config.forecast_env_wrapper_enabled:
        raise NotImplementedError(
            "ForecastWrapper is not supported in the multi-agent env creator."
        )

    logger.info("ma_env_creator: instance_id=%s agents=%s",
                instance_id, env.possible_agents)
    return env
