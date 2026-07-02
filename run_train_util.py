"""Shared CLI helpers for the training drivers.

``run_train_ray.py`` and ``run_train_sb.py`` take the same single ``--trial``
YAML and both translate the loaded ``TrialConfig`` into the legacy argparse
Namespace that ``log_startup_banner`` and related helpers were authored
against. Keeping the CLI parser and that adapter here stops the two drivers
from drifting.
"""

from __future__ import annotations

import argparse
from argparse import Namespace

from adv_building_gym.config.trial_config import TrialConfig


def parse_cli_args(description: str) -> argparse.Namespace:
    """Parse the shared training CLI (``--trial`` + ``--cpu``).

    Both drivers run on SLURM with a GPU by default and accept ``--cpu`` for
    a CPU-only smoke test, so the flag is identical across them; only the
    top-level ``description`` differs (SB3's names Stable-Baselines3).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--trial", type=str, required=True,
        help="Path to trial config YAML (e.g. configs/trial_cfgs/trial_cfg_1.yaml)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Smoke-test mode: bypass the GPU requirement and run on CPU.",
    )
    return parser.parse_args()


def trial_to_args_namespace(trial: TrialConfig) -> Namespace:
    """Adapt a loaded ``TrialConfig`` to the legacy argparse Namespace.

    The startup banner and its helpers expect an argparse Namespace; this
    preserves that interface for both drivers without re-plumbing every
    helper.
    """
    return Namespace(
        algorithm=trial.algorithm,
        episodes=trial.training_param_config.max_episodes_to_run,
        seed=trial.seed,
        metric=trial.metric,
        # Single cadence knob: eval + checkpoint share
        # training_params.common.evaluation.interval.
        checkpoint_frequency_iterations=trial.training_param_config.evaluation.interval,
        log_trajectories=trial.log_trajectories,
        num_envs=trial.num_envs,
        grad_train=trial.grad_train,
        trial_name=trial.trial_name,
        trial_path=str(trial.source_path) if trial.source_path else None,
    )
