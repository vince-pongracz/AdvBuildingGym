"""Ray RLlib training script for AdvBuildingGym environment.

Single CLI entry point: ``--trial <trial_cfg.yaml>``. The trial config
bundles algorithm + run control + env topology + training hyperparameters
+ data/reward/infra schedules. ``adv_building_gym.config.TrialConfig.load``
reads it once and resolves every sub-config so downstream modules receive
already-loaded objects.
"""

import os
import sys
import time
import datetime
import logging
from argparse import Namespace
from pathlib import Path

import json
import argparse
import torch

import ray
from ray import tune

from ray.tune import CLIReporter
from ray.tune.registry import register_env

from adv_building_gym.ray.utils.warning_filters import setup_warning_filters
from adv_building_gym.config.trial_config import TrialConfig
from adv_building_gym.ray.env_creator import adv_building_env_creator
from adv_building_gym.ray.training import common_model_setup, select_model
from adv_building_gym._common.json_encoder import CustomJSONEncoder
from adv_building_gym._common.rng_service import RngService
from adv_building_gym._common.resource_check_util import SlurmResources
from adv_building_gym.ray.utils.ray_utils import make_trial_dirname_creator
from adv_building_gym._common.startup_log import log_startup_banner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger("main")

# Environment variables to control Ray/RLlib behavior (must be set before ray.init).
# RAY_AIR_NEW_OUTPUT=0 keeps tune.RunConfig(progress_reporter=CLIReporter(...)) honoured.
# TUNE_DISABLE_STRICT_METRIC_CHECKING allows evaluation_interval > 1 without crashing on
# iterations that skip eval.
RUNTIME_ENV_VARS = {
    "PYTHONWARNINGS": "ignore::DeprecationWarning,ignore::UserWarning",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "RAY_METRICS_SERVICE_ENABLED": "0",
    "RAY_event_stats": "0",
    "RAY_DEDUP_LOGS": "0",
    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    "RAY_AIR_NEW_OUTPUT": "0",
    "TUNE_DISABLE_STRICT_METRIC_CHECKING": "1",
    "RAY_COLOR_PREFIX": os.environ.get("RAY_COLOR_PREFIX", ""),
    "TERM": os.environ.get("TERM", ""),
}
os.environ.update(RUNTIME_ENV_VARS)

# Apply warning filters in the main process
setup_warning_filters()

logger.info("Runtime environment variables for Ray workers: %s", RUNTIME_ENV_VARS)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _parse_cli_args() -> argparse.Namespace:
    """The trial config is the only input — every run parameter lives in it."""
    parser = argparse.ArgumentParser(
        description=(
            "Train an RL agent on AdvBuildingGym. The trial YAML "
            "bundles algorithm, env topology, hyperparameters, and schedules."
        ),
    )
    parser.add_argument(
        "--trial", type=str, required=True,
        help="Path to trial config YAML (e.g. configs/trial_cfgs/trial_cfg_1.yaml)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Smoke-test mode: bypass the GPU requirement and run learner on CPU.",
    )
    return parser.parse_args()


def _trial_to_args_namespace(trial: TrialConfig) -> Namespace:
    """Build a Namespace mirroring the legacy CLI args.

    The startup banner and helpers were authored against an argparse
    Namespace; this preserves that interface without re-plumbing every
    helper.
    """
    return Namespace(
        algorithm=trial.algorithm,
        episodes=trial.training_param_config.max_episodes_to_run,
        seed=trial.seed,
        metric=trial.metric,
        checkpoint_frequency_episodes=trial.checkpoint_frequency_episodes,
        log_trajectories=trial.log_trajectories,
        num_envs=trial.num_envs,
        grad_train=trial.grad_train,
        trial_name=trial.trial_name,
        trial_path=str(trial.source_path) if trial.source_path else None,
    )


# ---------------------------------------------------------------------------
# Ray initialisation
# ---------------------------------------------------------------------------

def _init_ray(seed: int, cpu_only: bool = False) -> SlurmResources:
    """Resolve SLURM resources, init Ray, and bring up the RngService actor."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus = int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else 2

    if cpu_only:
        gpus = 0
        logger.warning("CPU-only smoke-test mode: running learner on CPU (--cpu).")
    else:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            gpus = len([x for x in cuda_visible.split(",") if x.strip() != ""])
            if not torch.cuda.is_available():
                logger.error(
                    "SLURM allocated GPUs (CUDA_VISIBLE_DEVICES=%s) but PyTorch "
                    "cannot access CUDA. Check driver/CUDA toolkit setup.",
                    cuda_visible,
                )
                sys.exit(1)
        else:
            logger.error(
                "No GPU allocated (CUDA_VISIBLE_DEVICES is not set). "
                "Training requires a GPU — submit with --gres=gpu:1, "
                "or pass --cpu for a CPU-only smoke test.",
            )
            sys.exit(1)

    slurm_resources = SlurmResources(num_cpus=cpus, num_gpus=gpus)
    logger.info(
        "Training on device: %s (%d GPU(s))",
        "cpu" if gpus == 0 else "cuda", slurm_resources.num_gpus,
    )

    logger.info(
        "Initializing Ray with cpus=%s gpus=%s (from SLURM/CUDA env)",
        slurm_resources.num_cpus, slurm_resources.num_gpus,
    )
    ray.init(
        num_cpus=slurm_resources.num_cpus,
        num_gpus=slurm_resources.num_gpus,
        ignore_reinit_error=True,
        runtime_env={"env_vars": RUNTIME_ENV_VARS},
        logging_level=logging.INFO,
    )

    # Centralised RNG service as a Ray Named Actor. Must be initialised
    # AFTER ray.init() so the actor can be deployed.
    RngService.initialize(seed)

    return slurm_resources


# ---------------------------------------------------------------------------
# Algorithm config + checkpoint cadence
# ---------------------------------------------------------------------------

def _build_algo_config(args, trial: TrialConfig, slurm_resources, exec_date_dt):
    """Assemble the RLlib algorithm config and return ``(algo_config, param_space)``."""
    algo_config = select_model(
        algorithm=trial.algorithm,
        episode_length=trial.env_config.EPISODE_LENGTH,
        training_config=trial.training_param_config,
    )
    algo_config = common_model_setup(
        config=algo_config,
        training_config=trial.training_param_config,
        slurm_resources=slurm_resources,
        env_config=trial.env_config,
        metrics_base_dir="ep_metrics",
        data_combinator=trial.data_combinator,
        log_trajectories=trial.log_trajectories,
        reward_schedule_manager=trial.reward_manager,
        infra_combinator=trial.infra_combinator,
        statesource_combinator=trial.statesource_combinator,
        exploration_reset=trial.exploration_reset,
        exec_date=exec_date_dt,
        trial_name=trial.trial_name,
    )
    param_space = algo_config.to_dict()

    with open("param_space.json", "w", encoding="utf-8") as f:
        json.dump(param_space, f, cls=CustomJSONEncoder, indent=4)

    return algo_config, param_space


def _checkpoint_iterations(trial: TrialConfig) -> int:
    """Translate ``checkpoint_frequency_episodes`` into RLlib training iterations.

    train_batch_size_per_learner drives the timesteps RLlib processes per
    iteration but means different things per algorithm:
      PPO       — ppo_episodes_per_iteration * EPISODE_LENGTH (on-policy batch)
      SAC       — sac_replay_batch_size (off-policy replay sample)
      DreamerV3 — batch_size_B * batch_length_T (world-model training batch)
    """
    timesteps_per_episode = trial.env_config.EPISODE_LENGTH
    timesteps_per_iteration = 0.0

    if trial.algorithm == "ppo":
        timesteps_per_iteration = trial.training_param_config.ppo_episodes_per_iteration * trial.env_config.EPISODE_LENGTH
    if trial.algorithm == "sac":
        timesteps_per_iteration = trial.training_param_config.sac_replay_batch_size
    if trial.algorithm == "dreamerv3":
        timesteps_per_iteration = (
            trial.training_param_config.dreamerv3_batch_size_B
            * trial.training_param_config.dreamerv3_batch_length_T
        )

    if timesteps_per_iteration != 0.0:
        iters = max(1, int(
            (trial.checkpoint_frequency_episodes * timesteps_per_episode) / timesteps_per_iteration
        ))
    else:
        iters = 10
    
    logger.info(
        "Checkpoint configuration: every %d iterations (~%d episodes), metric=%s",
        iters, trial.checkpoint_frequency_episodes, trial.metric,
    )

    return iters


# ---------------------------------------------------------------------------
# Tuner construction
# ---------------------------------------------------------------------------

def _build_progress_reporter(algorithm: str) -> CLIReporter:
    """CLIReporter showing the algorithm-specific learner stats.

    Requires RAY_AIR_NEW_OUTPUT=0 — otherwise Tune's AIR output ignores
    ``metric_columns`` and renders its own default tables.
    """
    if algorithm == "ppo":
        algo_cols = {
            "learners/default_policy/policy_loss": "PolLoss",
            "learners/default_policy/vf_loss": "VfLoss",
            "learners/default_policy/mean_kl_loss": "KL",
            "learners/default_policy/mean_entropy": "Ent",
        }
    elif algorithm == "sac":
        algo_cols = {
            "learners/default_policy/critic_loss": "QLoss",
            "learners/default_policy/actor_loss": "PiLoss",
            "learners/default_policy/alpha_value": "Alpha",
            "learners/default_policy/td_error_mean": "TDErr",
        }
    elif algorithm == "dreamerv3":
        # TODO VP: Verify these metric paths against an actual DreamerV3
        # result.json (per memory feedback_rllib_new_api_stack.md). DreamerV3
        # learner logs world-model / actor / critic losses with names that
        # depend on the installed Ray version.
        algo_cols = {
            "learners/default_policy/WORLD_MODEL_total_loss": "WMLoss",
            "learners/default_policy/ACTOR_loss": "PiLoss",
            "learners/default_policy/CRITIC_L_total": "VfLoss",
        }
    else:
        algo_cols = {}

    return CLIReporter(
        metric_columns={
            "training_iteration": "Iter",
            "env_runners/num_episodes_lifetime": "Episodes",
            "time_total_s": "Time",
            "evaluation/env_runners/episode_return_mean": "EpReturnMean",
            "evaluation/env_runners/reward_rate": "RewardRate",
            "evaluation/env_runners/achieved_reward": "AchievedReward",
            **algo_cols,
        },
        max_report_frequency=30, # Report every 30 seconds
        print_intermediate_tables=True,
    )


def _build_tuner(trial: TrialConfig, metric: str, param_space, run_name, storage_path, checkpoint_freq_iterations, trial_name: str | None = None):
    """Build the ``tune.Tuner`` for the chosen algorithm."""
    stop_criteria = {
        # New API stack: lifetime episodes (1 episode = 1 day at 5-min control step).
        "env_runners/num_episodes_lifetime": trial.training_param_config.max_episodes_to_run,
    }
    progress_reporter = _build_progress_reporter(trial.algorithm)

    # Ray's algorithm registry is case-sensitive: "PPO"/"SAC" are all-caps,
    # but DreamerV3 is mixed-case — see ray.rllib.algorithms.registry.
    trainable_name = {"ppo": "PPO", "sac": "SAC", "dreamerv3": "DreamerV3"}[trial.algorithm]

    return tune.Tuner(
        trainable_name,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            reuse_actors=True,
            max_concurrent_trials=1,
            # Metrics (auto-prefixed with env_runners/):
            #   - "episode_return_mean"  (default RLlib metric)
            #   - "achieved_reward" (custom: sum of rewards per episode)
            #   - "reward_rate"     (custom: achieved/max possible reward)
            metric=metric,
            mode="max",
            trial_dirname_creator=make_trial_dirname_creator(trial_name),
        ),
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=storage_path,
            stop=stop_criteria,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=checkpoint_freq_iterations,
                num_to_keep=3,
                checkpoint_score_attribute=metric,
                checkpoint_score_order="max",
            ),
            progress_reporter=progress_reporter,
            verbose=2, # 0=silent, 1=less, 2=default, 3=verbose
        ),
    )


# ---------------------------------------------------------------------------
# Result post-processing
# ---------------------------------------------------------------------------

def _log_best_result(results, metric: str, storage_path: str) -> None:
    """Log the best trial's path, checkpoint, and key metrics."""
    try:
        best_result = results.get_best_result(metric=metric, mode="max")
        if best_result:
            logger.info("=" * 70)
            logger.info("Best trial results:")
            logger.info("  Trial directory: %s", best_result.path)
            if best_result.checkpoint:
                logger.info("  Best checkpoint: %s", best_result.checkpoint.path)
            else:
                logger.warning("  No checkpoint available for best trial")
            logger.info("  Storage path: %s", storage_path)
            logger.info("=" * 70)

        if best_result and best_result.metrics:
            env_runners_metrics = best_result.metrics.get("env_runners", {})
            metrics_to_log = {
                "episode_return_mean": env_runners_metrics.get("episode_return_mean"),
                "achieved_reward": env_runners_metrics.get("achieved_reward"),
                "reward_rate": env_runners_metrics.get("reward_rate"),
            }
            metric_key = metric.split("/")[-1]
            opt_value = metrics_to_log.get(metric_key)
            if opt_value is not None:
                logger.info("Optimized metric (%s): %.4f", metric, opt_value)
            for name, value in metrics_to_log.items():
                if value is not None and name != metric_key:
                    logger.info("  %s: %.4f", name, value)
        else:
            logger.warning("No best result or metrics available")
    except Exception as e:
        logger.error("Error retrieving best result: %s", str(e))


def _dump_all_results(results) -> None:
    """Write each trial's config + metrics + path to ``result_<i>.json``."""
    for i, res in enumerate(results._results):
        config_data = {}
        if res.config is not None:
            try:
                config_data = res.config.copy() if hasattr(res.config, "copy") else dict(res.config)
                if "_rl_module_spec" in config_data:
                    del config_data["_rl_module_spec"]
            except (TypeError, ValueError) as e:
                logger.warning("Could not serialize config for result %d: %s", i, e)
                config_data = {"error": str(e)}
        dump = {
            "config": config_data,
            "error": getattr(res, "error", None),
            "metrics": res.metrics if res.metrics is not None else {},
            "path": res.path,
        }
        with open(f"result_{i}.json", "w", encoding="utf-8") as f:
            json.dump(dump, f, cls=CustomJSONEncoder, indent=4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse the trial path, load all configs upfront, and run training."""
    cli_args = _parse_cli_args()
    trial = TrialConfig.load(cli_args.trial)
    logger.info(
        "Trial '%s' loaded from %s", trial.trial_name, trial.source_path,
    )

    # Initialise singleton component instances exactly once (driver only).
    # Triggers CSV parsing here; Ray workers go through factory methods.
    trial.env_config.init_singletons()

    # Auto-prefix metric so it points at the eval env_runners by default.
    metric = trial.metric
    if not metric.startswith("evaluation/"):
        metric = f"evaluation/env_runners/{metric}"

    args = _trial_to_args_namespace(trial)
    args.metric = metric  # banner uses the resolved metric

    slurm_resources = _init_ray(trial.seed, cpu_only=cli_args.cpu)

    exec_date_dt = datetime.datetime.now()
    exec_date = exec_date_dt.strftime("%Y%m%d_%H%M%S")
    run_name = f"{trial.algorithm}_seed{trial.seed}_{exec_date}"
    storage_path = os.path.abspath(f"models/{trial.trial_name}/ray/{trial.algorithm}")
    os.makedirs(storage_path, exist_ok=True)

    env_creator_config = {
        "env_config": trial.env_config,
        "data_combinator": trial.data_combinator,
        "reward_schedule_manager": trial.reward_manager,
    }
    register_env(
        "AdvBuilding",
        lambda cfg: adv_building_env_creator({**env_creator_config, **cfg}),
    )

    _, algo_cfg_param_space = _build_algo_config(args, trial, slurm_resources, exec_date_dt)
    checkpoint_freq_iterations = _checkpoint_iterations(trial)
    tuner = _build_tuner(
        trial, metric, algo_cfg_param_space, run_name, storage_path, checkpoint_freq_iterations,
        trial_name=trial.trial_name,
    )

    experiment_path = os.path.join(storage_path, run_name)
    log_startup_banner(
        args=args,
        env_config=trial.env_config,
        training_param_config=trial.training_param_config,
        reward_manager=trial.reward_manager,
        data_combinator=trial.data_combinator,
        infra_combinator=trial.infra_combinator,
        slurm_resources=slurm_resources,
        run_name=run_name,
        experiment_path=experiment_path,
        storage_path=storage_path,
        seed=trial.seed,
        exec_date=exec_date_dt,
    )

    logger.info("Starting tuner.fit() for: %s", run_name)
    logger.info("Training progress will be displayed below:")
    logger.info("=" * 70)

    t0 = time.time()
    results = tuner.fit()
    elapsed_time = (time.time() - t0) / 60
    logger.info("=" * 70)
    logger.info("Tuner/training finished in %.2f min", elapsed_time)
    logger.info("=" * 70)

    _log_best_result(results, metric, storage_path)
    _dump_all_results(results)

    ray.shutdown()
    logger.info("Script completed")


if __name__ == "__main__":
    main()

# Usage examples:
# On slurm: sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1.yaml
#
# python run_train_ray.py --trial configs/trial_cfgs/trial_cfg_1.yaml
