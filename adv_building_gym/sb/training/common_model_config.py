"""Algorithm-independent SB3 setup: SLURM resource resolution, train/eval VecEnv construction,
TensorBoard paths, and the callback list (metrics + schedules + checkpoint + eval).
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from adv_building_gym.config.trial_config import TrialConfig
from adv_building_gym._common.resource_check_util import SlurmResources

from adv_building_gym.sb.env_creator import build_vec_env
from adv_building_gym.sb.callbacks import (
    SBBestCheckpointCallback,
    SBEpisodeMetricsCallback,
    SBEvalStateActionCallback,
    SBIterTimingCallback,
    make_infra_schedule_callback,
    make_reward_switch_callback,
    make_statesource_schedule_callback,
    wrap_eval_callback_with_timer,
)
from adv_building_gym.config.rewards.reward_schedule_manager import RewardScheduleMode

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
    """Per-run directory tree under ``models/<trial>/sb3/<algo>/`` (mirrors the Ray layout)."""
    # absolute path so the banner emits full paths and TB finds the event files regardless of CWD
    base = (Path("models") / trial.trial_name / "sb3" / trial.algorithm / run_name).resolve()
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
    eval_trajectories_root: str,
) -> CallbackList:
    """Assemble training callbacks in Ray's composition order: metrics → schedule swaps
    (reward → infra → statesource) → best-by-metric checkpoint → eval callback."""
    callbacks: List[BaseCallback] = [
        SBIterTimingCallback(verbose=0),
        SBEpisodeMetricsCallback(verbose=1),
    ]

    # Data-variant selection is env-side (training follows the combinator cadence, eval draws
    # a fresh random variant), so no data-schedule callback here. The other three schedules
    # (reward/infra/statesource) use callbacks; their eval VecEnv follows training.

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

    # hand-rolled eval callback (replaces SB3's EvalCallback): owns the eval loop, saves
    # best_model.zip by mean reward, emits eval/* scalars + per-iter TB sub-run (parity with
    # Ray's EvalStateActionCallback). Wrapped with the iter-timing helper.
    eval_callback = SBEvalStateActionCallback(
        eval_env=eval_env,
        best_model_save_path=paths.best_dir,
        eval_trajectories_root=eval_trajectories_root,
        eval_freq=eval_freq_per_env,
        n_eval_episodes=n_eval_episodes,
        trial_name=trial.trial_name,
        deterministic=True,
        verbose=1,
    )
    wrap_eval_callback_with_timer(eval_callback)
    callbacks.append(eval_callback)

    callbacks.append(
        SBBestCheckpointCallback(
            checkpoint_dir=paths.checkpoint_dir,
            metric=trial.metric,
            checkpoint_frequency_iterations=trial.training_param_config.evaluation_interval,
            episode_return_mean_window=trial.training_param_config.episode_return_mean_window,
            num_to_keep=3,
            verbose=1,
        )
    )

    return CallbackList(callbacks)


def _resolve_eval_trajectories_root(
    metrics_base_dir: str, exec_date: datetime.datetime,
) -> str:
    """Mirror Ray's path computation ([ray/training/common_model_config.py:59-69](adv_building_gym/ray/training/common_model_config.py#L59-L69))."""
    job_suffix = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    run_dir_name = f"{exec_date.strftime('%Y%m%d_%H%M%S')}_{job_suffix}"
    return os.path.join(
        os.path.abspath(metrics_base_dir), "eval_trajectories", run_dir_name,
    )


def sb_common_model_setup(
    trial: TrialConfig,
    *,
    cpu_only: bool,
    exec_date: datetime.datetime | None = None,
) -> tuple[SlurmResources, SBRuntimePaths, "object", "object", "CallbackList"]:
    """Top-level setup for run_train_sb.py → (slurm, paths, train_vec, eval_vec, callbacks).
    The model is built separately by ``sb_select_model`` (so resources log before GPU alloc).
    """
    slurm = resolve_sb_resources(cpu_only=cpu_only)

    if exec_date is None:
        exec_date = datetime.datetime.now()
    exec_date_str = exec_date.strftime("%Y%m%d_%H%M%S")
    run_name = f"{trial.algorithm}_seed{trial.seed}_{exec_date_str}"
    paths = make_runtime_paths(trial, run_name)

    # eval is single-process (DummyVecEnv) for clean per-episode accounting
    train_vec = build_vec_env(
        trial, num_envs=trial.num_envs, seed=trial.seed, role="train",
    )
    eval_vec = build_vec_env(
        trial, num_envs=1, seed=trial.seed + 10_000, role="eval", force_dummy=True,
    )

    # Eval cadence: every ``evaluation_interval`` episodes (the single shared
    # cadence knob), matching SBBestCheckpointCallback so each checkpoint lands
    # on a fresh-eval round. ``eval_freq`` is compared against ``n_calls``, which
    # SB3 increments once per ``env.step()`` (i.e. once per *vec*-step = one
    # per-env step). evaluation_interval episodes across all envs complete after
    # ``evaluation_interval * EPISODE_LENGTH / num_envs`` vec-steps, so convert
    # episodes -> per-env steps (* EPISODE_LENGTH) and account for the parallel
    # envs (/ num_envs). Same semantics as SB3's stock EvalCallback, whose docs
    # likewise advise ``eval_freq // n_envs`` for vectorised envs.
    eval_freq_per_env = max(
        1,
        (trial.training_param_config.evaluation_interval
         * trial.env_config.EPISODE_LENGTH)
        // trial.num_envs,
    )

    eval_trajectories_root = _resolve_eval_trajectories_root("ep_metrics", exec_date)

    callback_list = build_callback_list(
        trial,
        paths=paths,
        eval_env=eval_vec,
        eval_freq_per_env=eval_freq_per_env,
        n_eval_episodes=trial.training_param_config.evaluation_duration,  # parity with Ray's evaluation_duration
        num_envs=trial.num_envs,
        eval_trajectories_root=eval_trajectories_root,
    )

    return slurm, paths, train_vec, eval_vec, callback_list
