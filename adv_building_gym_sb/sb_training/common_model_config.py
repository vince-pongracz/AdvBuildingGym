"""Common setup helpers for SB3 training.

This module wires together the bits that don't depend on the algorithm
choice: SLURM-derived resource resolution, eval/train VecEnv construction,
TensorBoard logger pointing at the right path, and assembling the callback
list (episode metrics + schedules + checkpoint + EvalCallback).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

from adv_building_gym import TrialConfig
from adv_building_gym.utils import SlurmResources

from adv_building_gym_sb.env_creator import build_vec_env
from adv_building_gym_sb.callbacks import (
    SBBestCheckpointCallback,
    SBEpisodeMetricsCallback,
    SBIterTimingCallback,
    make_data_schedule_callback,
    make_infra_schedule_callback,
    make_reward_switch_callback,
    make_statesource_schedule_callback,
    wrap_eval_callback_with_timer,
)
from adv_building_gym.config.reward_schedule_manager import RewardScheduleMode

logger = logging.getLogger(__name__)


@dataclass
class SBRuntimePaths:
    """Resolved on-disk locations for the run."""
    model_dir: str       # final model + summary.json
    checkpoint_dir: str  # best-by-metric checkpoints
    best_dir: str        # SB3 EvalCallback's best_mean_reward model
    log_dir: str         # TensorBoard event files


def resolve_sb_resources(*, cpu_only: bool) -> SlurmResources:
    """SLURM resource detection mirroring run_train_ray._init_ray.

    Hard-requires a GPU unless ``cpu_only`` is True (the ``--cpu`` smoke
    escape hatch). CPU count is informational for the SB3 driver — we
    don't shard learners across processes — but is surfaced so the
    startup log matches the Ray driver's accounting.
    """
    slurm_cpus_env = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus = int(slurm_cpus_env) if slurm_cpus_env and slurm_cpus_env.isdigit() else 2

    if cpu_only:
        logger.warning("CPU-only smoke-test mode: running SB3 on CPU (--cpu).")
        return SlurmResources(num_cpus=cpus, num_gpus=0)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible:
        logger.error(
            "No GPU allocated (CUDA_VISIBLE_DEVICES is not set). "
            "SB3 training requires a GPU — submit with --gres=gpu:1, "
            "or pass --cpu for a CPU-only smoke test.",
        )
        sys.exit(1)

    gpus = len([x for x in cuda_visible.split(",") if x.strip()])
    if not torch.cuda.is_available():
        logger.error(
            "SLURM allocated GPUs (CUDA_VISIBLE_DEVICES=%s) but PyTorch "
            "cannot access CUDA. Check driver / CUDA toolkit setup.",
            cuda_visible,
        )
        sys.exit(1)

    return SlurmResources(num_cpus=cpus, num_gpus=gpus)


def make_runtime_paths(trial: TrialConfig, run_name: str) -> SBRuntimePaths:
    """Build the per-run directory tree under ``models/<trial>/sb3/<algo>/``.

    Mirrors the Ray driver's storage layout (``models/<trial>/ray/<algo>/``)
    so downstream tooling (TB launch scripts, checkpoint finders) can
    treat the two frameworks symmetrically.
    """
    base = Path("models") / trial.trial_name / "sb3" / trial.algorithm / run_name
    base.mkdir(parents=True, exist_ok=True)

    paths = SBRuntimePaths(
        model_dir=str(base),
        checkpoint_dir=str(base / "checkpoints"),
        best_dir=str(base / "best"),
        log_dir=str(base / "tb"),
    )
    for p in (paths.checkpoint_dir, paths.best_dir, paths.log_dir):
        Path(p).mkdir(parents=True, exist_ok=True)
    return paths


def build_callback_list(
    trial: TrialConfig,
    *,
    paths: SBRuntimePaths,
    eval_env,
    eval_freq_per_env: int,
    n_eval_episodes: int,
    num_envs: int,
) -> CallbackList:
    """Assemble all training callbacks in the same order Ray composes them.

    Order matters: episode metrics first (so per-component reward/cost
    totals are computed before any consumer reads them), then schedule
    swaps (data → reward → infra → statesource — matches the Ray
    composition order in ``register_callbacks``), then the best-by-metric
    checkpoint, then SB3's stock EvalCallback.
    """
    callbacks: List[BaseCallback] = [
        SBIterTimingCallback(verbose=0),
        SBEpisodeMetricsCallback(verbose=1),
    ]

    # Data variant scheduling — always-on (empty combinator is a no-op).
    # Eval VecEnv follows training so the eval signal describes the regime
    # the policy is currently being trained on. Same for the other three
    # schedules below.
    if trial.data_combinator is not None and trial.data_combinator.variants:
        callbacks.append(
            make_data_schedule_callback(
                trial.data_combinator,
                num_env_runners=num_envs,
                eval_env=eval_env,
            )
        )

    if (trial.reward_manager is not None
            and trial.reward_manager.mode is not RewardScheduleMode.OFF):
        callbacks.append(
            make_reward_switch_callback(
                trial.reward_manager,
                num_env_runners=num_envs,
                exploration_reset=trial.exploration_reset,
                eval_env=eval_env,
            )
        )

    if trial.infra_combinator is not None and trial.infra_combinator.is_enabled():
        callbacks.append(
            make_infra_schedule_callback(
                trial.infra_combinator,
                num_env_runners=num_envs,
                exploration_reset=trial.exploration_reset,
                eval_env=eval_env,
            )
        )

    if (trial.statesource_combinator is not None
            and trial.statesource_combinator.is_enabled()):
        callbacks.append(
            make_statesource_schedule_callback(
                trial.statesource_combinator,
                num_env_runners=num_envs,
                exploration_reset=trial.exploration_reset,
                eval_env=eval_env,
            )
        )

    # SB3's stock EvalCallback handles deterministic eval + best_mean_reward
    # tracking. We layer our own metric-aware checkpoint callback on top so
    # achieved_reward / reward_rate selections work too. The eval call is
    # wrapped with the iter-timing helper so timers/eval_s is recorded.
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=paths.best_dir,
        log_path=paths.log_dir,
        eval_freq=eval_freq_per_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    wrap_eval_callback_with_timer(eval_callback)
    callbacks.append(eval_callback)

    callbacks.append(
        SBBestCheckpointCallback(
            checkpoint_dir=paths.checkpoint_dir,
            metric=trial.metric,
            checkpoint_frequency_episodes=trial.checkpoint_frequency_episodes,
            num_to_keep=3,
            verbose=1,
        )
    )

    return CallbackList(callbacks)


def sb_common_model_setup(
    trial: TrialConfig,
    *,
    cpu_only: bool,
) -> tuple[SlurmResources, SBRuntimePaths, "object", "object", "CallbackList"]:
    """Top-level setup function used by run_train_sb.py.

    Returns:
        (slurm_resources, runtime_paths, train_vec_env, eval_vec_env, callback_list)
        — the model itself is built separately by ``sb_select_model`` so
        the caller can log resources before allocating GPU memory.
    """
    slurm = resolve_sb_resources(cpu_only=cpu_only)

    import datetime
    exec_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{trial.algorithm}_seed{trial.seed}_{exec_date}"
    paths = make_runtime_paths(trial, run_name)

    # Eval is always single-process (DummyVecEnv) for clean per-episode
    # accounting and reproducible deterministic rollouts.
    train_vec = build_vec_env(
        trial, num_envs=trial.num_envs, seed=trial.seed, role="train",
    )
    eval_vec = build_vec_env(
        trial, num_envs=1, seed=trial.seed + 10_000, role="eval", force_dummy=True,
    )

    # Eval cadence: roughly every ``checkpoint_frequency_episodes`` episodes,
    # divided by num_envs since EvalCallback's eval_freq is per env.
    eval_freq_per_env = max(
        1,
        (trial.checkpoint_frequency_episodes * trial.env_config.EPISODE_LENGTH)
        // max(1, trial.num_envs),
    )

    callback_list = build_callback_list(
        trial,
        paths=paths,
        eval_env=eval_vec,
        eval_freq_per_env=eval_freq_per_env,
        n_eval_episodes=2,  # parity with Ray's evaluation_duration=2
        num_envs=trial.num_envs,
    )

    return slurm, paths, train_vec, eval_vec, callback_list
