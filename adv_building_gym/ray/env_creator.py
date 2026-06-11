"""Env creator factory for Ray RLlib/Tune — builds AdvBuildingGym instances per call."""

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
    """Merge the static config with RLlib's EnvContext, preserving ``worker_index`` /
    ``vector_index`` (a plain ``{**base, **cfg}`` drops them — they're EnvContext attrs)."""
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
    """Create a wrapped AdvBuildingGym (flat Box(-1, 1) action space) for Ray Tune.

    Builds FRESH components each call so parallel env_runners don't share state.
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

    # fresh per-env instances (independent state) from the YAML specs
    infras = env_config.create_infras()
    statesources = env_config.create_statesources()

    # Rewards always come from the RewardScheduleManager (mode=OFF returns all).
    reward_manager = config["reward_schedule_manager"]
    rewards = reward_manager.create_active_rewards()

    # RLlib indices (via merge_env_context): worker_index 0=local, 1..N=remote;
    # vector_index = sub-env slot within the worker
    worker_index = getattr(config, "worker_index", 0)
    vector_index = getattr(config, "vector_index", 0)
    instance_id = f"AdvBuildingGym_w{worker_index}_v{vector_index}"

    # DEBUG: confirm EnvContext metadata reached here ("MISSING" = dropped earlier)
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

    if env_config.hst.enabled:
        env = HistoryWrapper(
            env,
            tracked_keys=env_config.hst.tracked_keys,
            offsets=env_config.hst.offsets,
        )
        logger.info(
            "env_creator: HistoryWrapper enabled (tracked_keys=%s, offsets=%s)",
            list(env_config.hst.tracked_keys),
            list(env_config.hst.offsets),
        )

    if env_config.forecast.enabled:
        env = ForecastWrapper(env, forecast_steps=env_config.forecast.steps)
        logger.info(
            "env_creator: ForecastWrapper enabled (steps=%s)",
            list(env_config.forecast.steps),
        )

    return wrap_action_space(env)


def adv_building_ma_env_creator(config: dict):
    """Multi-agent (per-actuator) variant — like :func:`adv_building_env_creator` but builds a
    :class:`MultiAgentAdvBuildingGym` (rescale lives inside it, no flat-Box wrappers).

    Optional ``reward_partition`` (dict[agent_id → reward names]) splits reward components per agent;
    ``None`` routes the global reward to every agent (cooperative MARL).
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

    # DEBUG: confirm EnvContext metadata reached here ("MISSING" → os.getpid() fallback below)
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
    # TODO VP 2026.05.31.: Add eval mode to MA env and set it here when supported.

    if env_config.hst.enabled:
        # HistoryWrapper targets the single-agent Dict obs; not supported for the per-agent MA space
        raise NotImplementedError("HistoryWrapper is not supported in the multi-agent env creator.")

    if env_config.forecast.enabled:
        raise NotImplementedError("ForecastWrapper is not supported in the multi-agent env creator.")

    logger.info("ma_env_creator: instance_id=%s agents=%s",
                instance_id, env.possible_agents)
    return env
