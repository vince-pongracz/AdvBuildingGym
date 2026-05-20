"""SB3 environment factory.

Mirrors :func:`adv_building_gym.envs.env_creator.adv_building_env_creator`
but is shaped for ``SubprocVecEnv`` / ``DummyVecEnv``: returns a *thunk*
that constructs one fresh env when invoked inside the SB3 worker.

The wrapper chain matches the Ray side exactly so the policy sees an
identical observation/action space:

    AdvBuildingGym
      → HistoryWrapper            (if env_config.hst_env_wrapper_enabled)
      → ForecastWrapper           (if env_config.forecast_env_wrapper_enabled)
      → FlattenAction
      → RescaleAction(-1, 1)
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from adv_building_gym import TrialConfig
from adv_building_gym.envs import AdvBuildingGym
from adv_building_gym.envs.env_creator import wrap_action_space
from adv_building_gym.envs.forecast_wrapper import ForecastWrapper
from adv_building_gym.envs.history_wrapper import HistoryWrapper
from adv_building_gym.rewards import SumRewardAggregator

logger = logging.getLogger(__name__)

_env_instance_counter = itertools.count()


def make_sb_env_factory(
    trial: TrialConfig,
    *,
    rank: int,
    seed: int,
    role: str = "train",
) -> Callable[[], gym.Env]:
    """Return a thunk that builds one fresh wrapped env on call.

    Each thunk produces an independent component bundle so parallel
    SubprocVecEnv workers never share mutable state.

    Args:
        trial: Loaded TrialConfig (env topology, reward pool, schedules).
        rank: Index of this env in the VecEnv (used to disambiguate
            seed streams and instance ids).
        seed: Base seed; the env is reset with ``seed + rank`` to give
            each runner a distinct stream while remaining deterministic.
        role: ``"train"`` or ``"eval"``. Eval envs enable
            ``log_full_info`` so callbacks see ``info["state"]``.
    """

    env_config = trial.env_config
    reward_manager = trial.reward_manager
    data_combinator = trial.data_combinator

    def _init() -> gym.Env:
        infras = env_config.create_infras()
        statesources = env_config.create_statesources()
        rewards = reward_manager.create_active_rewards()

        instance_id = (
            f"AdvBuildingGymSB_{role}_w{os.getpid()}_v{next(_env_instance_counter)}_r{rank}"
        )

        env = AdvBuildingGym(
            infras=infras,
            statesources=statesources,
            rewards=rewards,
            env_config=env_config,
            data_combinator=data_combinator,
            reward_aggregator=SumRewardAggregator(),
            instance_id=instance_id,
        )

        if role == "eval":
            env.log_full_info = True

        if env_config.hst_env_wrapper_enabled:
            env = HistoryWrapper(
                env,
                tracked_keys=env_config.hst_env_wrapper_tracked_keys,
                offsets=env_config.hst_env_wrapper_offsets,
            )
        if env_config.forecast_env_wrapper_enabled:
            env = ForecastWrapper(env, forecast_steps=env_config.forecast_env_wrapper_steps)

        env = wrap_action_space(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(
    trial: TrialConfig,
    *,
    num_envs: int,
    seed: int,
    role: str = "train",
    force_dummy: bool = False,
) -> VecEnv:
    """Build a SB3 VecEnv (Dummy- or Subproc-) wrapped in VecMonitor.

    ``VecMonitor`` is what populates ``infos[i]["episode"]`` on episode
    completion — many SB3 callbacks rely on that key.

    Args:
        trial: Loaded trial config.
        num_envs: Number of parallel sub-envs.
        seed: Base seed used inside the thunks.
        role: ``"train"`` or ``"eval"``.
        force_dummy: When True, always use DummyVecEnv regardless of
            ``num_envs``. Useful for evaluation (single process, simpler
            tracebacks) and for the ``--cpu`` smoke path.
    """
    factories = [
        make_sb_env_factory(trial, rank=i, seed=seed, role=role) for i in range(num_envs)
    ]
    if force_dummy or num_envs <= 1:
        vec_env: VecEnv = DummyVecEnv(factories)
    else:
        vec_env = SubprocVecEnv(factories)
    return VecMonitor(vec_env)
