"""Env creator factory for Ray RLlib/Tune — builds AdvBuildingGym instances per call."""

import itertools
import logging
import os
import gymnasium

from adv_building_gym.components.rewards import SumRewardAggregator

from adv_building_gym.config.data.data_combinator import DataCombinator
from adv_building_gym.config.env.env_config import EnvConfig
from adv_building_gym.config.rewards.reward_schedule_manager import RewardScheduleManager
from adv_building_gym.core.env import AdvBuildingGym
from adv_building_gym.core.forecast_wrapper import ForecastWrapper
from adv_building_gym.core.history_wrapper import HistoryWrapper
from adv_building_gym.ray.ma_env import MultiAgentAdvBuildingGym
from adv_building_gym.core.wrappers import wrap_action_space

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
    """Create a wrapped AdvBuildingGym (flat Box(-1, 1) action space, flat Box obs) for Ray Tune.

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
            - ``eval_data_combinator``: Held-out eval DataCombinator; used
              instead of ``data_combinator`` when ``eval_mode`` is set, so the
              in-training evaluation rounds sample from the eval dataset.
            - ``log_full_info``: When True, enables deep-copy of named
              state into info["state"] each step (evaluation only).
            - ``eval_mode``: Set by Ray's evaluation env_config override; routes
              the env to ``eval_data_combinator`` and fresh-random per-episode
              sampling.
            - ``seed``: Base construction seed for training envs (trial ``seed:``).
            - ``eval_seed``: Base construction seed used instead of ``seed`` when
              ``eval_mode`` is set (trial ``eval_seed:``, which defaults to
              ``seed:``). Absent → falls back to ``seed``.

    Returns:
        Wrapped AdvBuildingGym with flat Box(-1, 1) action space and flat Box obs.
    """
    
    # NOTE VP 2026.07.05.: Observation flattening is env-side for ALL algorithms: the LAST (outermost)
    # wrapper is ``gymnasium.wrappers.FlattenObservation``, so no connector-side
    # flattener is needed (see ``common_model_setup(flatten_observations_env_side=True)``).
    # Flat feature order = the Dict space's key order (gymnasium flatten iterates
    # ``spaces.items()``) — NOT the sorted order RLlib's FlattenObservations connector
    # produced; checkpoints are only compatible with the mechanism they trained on.
    
    env_config: EnvConfig | None = config.get("env_config")
    if env_config is None:
        raise ValueError(
            "adv_building_env_creator: 'env_config' missing from creator config. "
            "The training driver must register the env with the active EnvConfig "
            "(e.g. env_creator_config={'env_config': active_config, ...})."
        )

    # fresh per-env instances (independent state) from the YAML specs
    # TODO noprio VP 2026.08.06.: Refactor these, so they are not called here but inside AdvBuildingGym.__init__
    # The env_config is already passed to the constructor...
    infras: list = env_config.create_infras()
    statesources: list = env_config.create_statesources()

    # Rewards always come from the RewardScheduleManager (mode=OFF returns all).
    reward_manager: RewardScheduleManager = config["reward_schedule_manager"]
    rewards: list = reward_manager.create_active_rewards()

    # RLlib indices (via merge_env_context): worker_index 0=local, 1..N=remote;
    # vector_index = sub-env slot within the worker
    worker_index: int = getattr(config, "worker_index", 0)
    vector_index: int = getattr(config, "vector_index", 0)
    instance_id: str = f"AdvBuildingGym_w{worker_index}_v{vector_index}"
    is_eval: bool = config.get("eval_mode", False)
    if is_eval:
        instance_id = f"Eval_{instance_id}"

    # DEBUG: confirm EnvContext metadata reached here ("MISSING" = dropped earlier)
    logger.info(
        "env_creator: worker_index=%s vector_index=%s recreated=%s num_workers=%s "
        "(cfg type=%s) → instance_id=%s",
        getattr(config, "worker_index", "MISSING"), getattr(config, "vector_index", "MISSING"),
        getattr(config, "recreated_worker", "?"), getattr(config, "num_workers", "?"),
        type(config).__name__, instance_id,
    )

    # Eval EnvRunners carry eval_mode=True (set via the evaluation_config env_config
    # override); they sample from the held-out eval combinator when one is supplied,
    # so the in-training eval rounds run on the eval dataset rather than the train one.
    data_combinator: DataCombinator | None = config.get("data_combinator")
    if is_eval and config.get("eval_data_combinator") is not None:
        data_combinator = config.get("eval_data_combinator")
        logger.info("env_creator: eval_mode — using eval data_combinator for %s", instance_id)

    env = AdvBuildingGym(
        env_config=env_config,
        statesources=statesources,
        infras=infras,
        rewards=rewards,
        data_combinator=data_combinator,
        reward_aggregator=SumRewardAggregator(),
        instance_id=instance_id,
        eval_mode=is_eval,
    )
    # Derive a unique seed per env instance. Eval EnvRunners take ``eval_seed`` (the trial's
    # `eval_seed:`, defaulting to `seed:`) so a training seed sweep leaves the eval episode
    # sequence fixed. This reset is the ONLY one that reaches an eval env's RNG: it sets
    # _has_seeded, after which AdvBuildingGym._maybe_reseed ignores every later reset seed in
    # eval_mode — including RLlib's own `seed + 1e6` — so RLlib cannot override the value here.
    base_seed = config.get("seed", 21)
    if is_eval:
        base_seed = config.get("eval_seed", base_seed)
    seed = base_seed + worker_index + 1000 * vector_index
    env.reset(seed=seed)
    logger.info("Env: instance_id=%s seed=%s (base=%s, %s)",
                instance_id, seed, base_seed, "eval_seed" if is_eval else "seed")
    
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

    env = wrap_action_space(env)

    # Env-side obs flattening for ALL algorithms; MUST stay the outermost wrapper
    # Required by DreamerV3, it reads env.single_observation_space directly
    # (Dict spaces have shape=None and crash do_symlog_obs).
    # Link: ray/rllib/algorithms/dreamerv3/utils/__init__.py (do_symlog_obs)
    # gymnasium's flatten_space/flatten both iterate the Dict space's spaces.items(),
    # so flat values keep the Dict's key order, aligned with the flat space bounds.
    # Link: gymnasium/spaces/utils.py (_flatten_dict / _flatten_space_dict)
    env = gymnasium.wrappers.FlattenObservation(env)
    logger.info(
        "env_creator: FlattenObservation applied (outermost; flat obs shape=%s)", env.observation_space.shape,
    )

    return env


def adv_building_ma_env_creator(config: dict):
    """Multi-agent (per-actuator) variant — like :func:`adv_building_env_creator` but builds a
    :class:`MultiAgentAdvBuildingGym` (rescale lives inside it, no flat-Box wrappers).

    Optional ``reward_partition`` (dict[agent_id → reward names]) splits reward components per agent;
    ``None`` routes the global reward to every agent (cooperative MARL).
    """
    env_config = config.get("env_config")
    if env_config is None:
        raise ValueError("adv_building_ma_env_creator: 'env_config' missing from creator config.")

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

    logger.info("ma_env_creator: instance_id=%s agents=%s", instance_id, env.possible_agents)
    return env
