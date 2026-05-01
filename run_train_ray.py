"""Ray RLlib training script for AdvBuildingGym environment.

Configures RL algorithms (PPO, SAC), manages distributed training with
automatic SLURM resource detection, handles checkpointing, and tracks multi-objective
reward metrics during training.
"""

import os
import sys
import time
import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import argparse
import torch

import ray
from ray import tune

from ray.tune import CLIReporter
from ray.tune.registry import register_env

# Import warning filter setup from utils
# The function is centrally defined in adv_building_gym/utils/warning_filters.py
# and is called in two places:
#   1. Here in the main process (before Ray starts)
#   2. In AdvBuildingGym.__init__() (runs in each Ray worker when env is created)
from adv_building_gym.utils import setup_warning_filters

# Trigger registration of the custom Gym IDs
from adv_building_gym import DataCombinator, EnvConfigManager
from adv_building_gym.config import EnvConfig, load_data_combinator_config
from adv_building_gym.config.reward_schedule_manager import RewardScheduleManager, RewardScheduleMode
from adv_building_gym.envs import adv_building_env_creator
from adv_building_gym.infra_combinator import InfraCombinator
from adv_building_gym.ray_training import common_model_setup, select_model
from adv_building_gym.config.training_param_config import TrainingParamConfig
from adv_building_gym.utils import (
    CustomJSONEncoder,
    RngService,
    SlurmResources,
    trial_dirname_creator,
)
from adv_building_gym.utils.startup_log import log_startup_banner

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # Override any existing logging configuration (e.g., from Ray/RLlib)
)
logger = logging.getLogger("main")

# Environment variables to control Ray/RLlib behavior (must be set before ray.init).
# Set in the local process and propagated to Ray workers via runtime_env.
# Note: PYTHONWARNINGS is comma-separated, not colon-separated.
# RAY_AIR_NEW_OUTPUT=0 keeps tune.RunConfig(progress_reporter=CLIReporter(...)) honoured;
#   the new AIR output (default in Ray >=2.7) ignores `metric_columns`.
#   Link: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.ProgressReporter.html
# TUNE_DISABLE_STRICT_METRIC_CHECKING allows evaluation_interval > 1 without crashing on
#   iterations that skip eval (the eval metric key is absent from non-eval results).
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
    """Build and parse the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", default="ppo", choices=["ppo", "sac"],
        help="RL algorithm to train (default: ppo)",
    )
    parser.add_argument(
        "--load-config", type=str, required=True,
        help="Path to env wrapper YAML to load (required, e.g., 'configs/env/env_test1_small.yaml'). "
             "The wrapper references separate infras / statesources / env_meta YAMLs.",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Total training episodes. Primary stopping criterion. "
             "Defaults to TrainingParamConfig.max_episodes_to_run.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--metric",
        type=str,
        default="reward_rate",
        choices=[
            # NOTE VP 2026.01.12: Diff between episode_return_mean and achieved_reward:
            # - achieved_reward: Custom metric from episode_callbacks.py
            #   sum(rewards) per episode, averaged across episodes in the CURRENT iteration only (~14 episodes)
            #   More responsive to recent performance changes
            # - episode_return_mean: RLlib built-in metric
            #   Same base calculation, but EMA-smoothed over last 25 episodes
            #   More stable, less sensitive to noise, better for long-term trends
            # - reward_rate: Custom metric = achieved_reward / max_possible_reward
            #   Normalized performance score in [0, 1]
            "episode_return_mean",
            "achieved_reward",
            "reward_rate",
        ],
        help="Metric to optimize during training (auto-prefixed with 'env_runners/')",
    )
    parser.add_argument(
        "--checkpoint-frequency-episodes",
        type=int,
        default=20,
        help="Checkpoint frequency in number of episodes (will be converted to training iterations)",
    )
    parser.add_argument(
        "--log-trajectories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-step trajectory JSON per episode (default: False)",
    )
    parser.add_argument(
        "--data-config", type=str, default=None,
        help="Path to data combinator YAML config (default: configs/data_scheduler/train_data_combinator_config.yaml)",
    )
    parser.add_argument(
        "--grad-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradual reward training (curriculum). When inactive, all rewards "
             "are active from start.",
    )
    parser.add_argument(
        "--reward-schedule", type=str, default=None,
        help="Path to reward schedule YAML config "
             "(default: configs/reward_cfg/reward_schedule_train.yaml)",
    )
    parser.add_argument(
        "--infra-schedule", type=str, default=None,
        help="Path to infra schedule YAML config "
             "(e.g. 'configs/infra_schedule/infra_schedule_train.yaml').",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class LoadedConfigs:
    """Bundle of configuration objects produced by ``_load_configs``."""
    training_param_config: TrainingParamConfig
    active_config: EnvConfig
    data_combinator: DataCombinator
    reward_manager: RewardScheduleManager
    infra_combinator: Optional[InfraCombinator]


def _load_configs(args: argparse.Namespace) -> LoadedConfigs:
    """Load env / data / reward / infra schedule configs and resolve seed + episodes."""
    training_param_config = TrainingParamConfig.from_yaml(
        Path(__file__).resolve().parent / "configs" / "training_param_config.yaml"
    )

    if args.seed is not None:
        training_param_config.seed = args.seed
    else:
        args.seed = training_param_config.seed

    logger.info("Loading env config from: %s", args.load_config)
    active_config = EnvConfigManager.load(args.load_config)

    data_combinator = load_data_combinator_config(
        cfg_yaml_path=args.data_config,
        seed_override=args.seed,
    )

    # Reward schedule. With --grad-train the manager uses its YAML mode
    # (gradual_add / iterate / random). Without it, mode is forced to OFF
    # so all configured rewards are active from the start.
    reward_schedule_path = args.reward_schedule or str(
        Path(__file__).resolve().parent / "configs" / "reward_cfg" / "reward_schedule_train.yaml"
    )
    reward_manager = RewardScheduleManager.from_yaml(reward_schedule_path)
    if not args.grad_train:
        reward_manager.mode = RewardScheduleMode.OFF
        logger.info("Gradual training disabled — all specified rewards active from start")
    else:
        logger.info(
            "Gradual training enabled: mode=%s, swap every %d iterations",
            reward_manager.mode, reward_manager.swap_every_n_iterations,
        )

    infra_combinator = None
    if args.infra_schedule:
        infra_combinator = InfraCombinator.from_yaml(
            args.infra_schedule, control_step=active_config.CONTROL_STEP,
        )
        logger.info(
            "Infrastructure schedule enabled: mode=%s, %d configs, "
            "swap every %d iterations",
            infra_combinator.mode, len(infra_combinator.config_paths),
            infra_combinator.swap_every_n_iterations,
        )

    if args.episodes is None:
        args.episodes = training_param_config.max_episodes_to_run
        logger.info("Using default: %d episodes", args.episodes)
    else:
        logger.info("Stopping after %d episodes", args.episodes)

    # Initialise singleton component instances exactly once (driver only).
    # Triggers CSV parsing here; Ray workers go through factory methods.
    active_config.init_singletons()

    if not args.metric.startswith("evaluation/"):
        args.metric = f"evaluation/env_runners/{args.metric}"

    return LoadedConfigs(
        training_param_config=training_param_config,
        active_config=active_config,
        data_combinator=data_combinator,
        reward_manager=reward_manager,
        infra_combinator=infra_combinator,
    )


# ---------------------------------------------------------------------------
# Ray initialisation
# ---------------------------------------------------------------------------

def _init_ray(seed: int) -> SlurmResources:
    """Resolve SLURM resources, init Ray, and bring up the RngService actor."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus = int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else 2

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
            "Training requires a GPU — submit with --gres=gpu:1.",
        )
        sys.exit(1)

    slurm_resources = SlurmResources(num_cpus=cpus, num_gpus=gpus)
    logger.info("Training on device: cuda (%d GPU(s) from SLURM)", slurm_resources.num_gpus)

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

def _build_algo_config(args, configs: LoadedConfigs, slurm_resources, exec_date_dt):
    """Assemble the RLlib algorithm config and return ``(algo_config, param_space)``."""
    algo_config = select_model(
        algorithm=args.algorithm,
        episode_length=configs.active_config.EPISODE_LENGTH,
        training_config=configs.training_param_config,
    )
    algo_config = common_model_setup(
        config=algo_config,
        training_config=configs.training_param_config,
        slurm_resources=slurm_resources,
        env_config=configs.active_config,
        metrics_base_dir="ep_metrics",
        data_combinator=configs.data_combinator,
        log_trajectories=args.log_trajectories,
        reward_schedule_manager=configs.reward_manager,
        infra_combinator=configs.infra_combinator,
        exec_date=exec_date_dt,
    )
    param_space = algo_config.to_dict()

    # Save parameter space for inspection.
    # NOTE: param_space stores both new and old API entries for backward compat —
    # the model dict has defaults; _model_config has the actual spec.
    with open("param_space.json", "w", encoding="utf-8") as f:
        json.dump(param_space, f, cls=CustomJSONEncoder, indent=4)

    return algo_config, param_space


def _checkpoint_iterations(args, configs: LoadedConfigs) -> int:
    """Translate ``--checkpoint-frequency-episodes`` into RLlib training iterations.

    train_batch_size_per_learner drives the timesteps RLlib processes per
    iteration but means different things per algorithm:
      PPO — ppo_episodes_per_iteration × EPISODE_LENGTH (on-policy batch)
      SAC — sac_replay_batch_size (off-policy replay sample)
    """
    timesteps_per_episode = configs.active_config.EPISODE_LENGTH
    if args.algorithm == "ppo":
        timesteps_per_iteration = (
            configs.training_param_config.ppo_episodes_per_iteration
            * configs.active_config.EPISODE_LENGTH
        )
    else:
        timesteps_per_iteration = configs.training_param_config.sac_replay_batch_size

    iters = max(1, int(
        (args.checkpoint_frequency_episodes * timesteps_per_episode) / timesteps_per_iteration
    ))
    logger.info(
        "Checkpoint configuration: every %d iterations (~%d episodes), metric=%s",
        iters, args.checkpoint_frequency_episodes, args.metric,
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


def _build_tuner(args, param_space, run_name, storage_path, checkpoint_freq_iterations):
    """Build the ``tune.Tuner`` for the chosen algorithm."""
    stop_criteria = {
        # New API stack: lifetime episodes (1 episode = 1 day at 5-min control step).
        "env_runners/num_episodes_lifetime": args.episodes,
    }
    progress_reporter = _build_progress_reporter(args.algorithm)

    return tune.Tuner(
        args.algorithm.upper(),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            reuse_actors=True,
            max_concurrent_trials=1,
            # Metric set via --metric CLI argument (auto-prefixed with env_runners/):
            #   - "episode_return_mean"  (default RLlib metric)
            #   - "achieved_reward" (custom: sum of rewards per episode)
            #   - "reward_rate"     (custom: achieved/max possible reward)
            metric=args.metric,
            mode="max",
            trial_dirname_creator=trial_dirname_creator,
        ),
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=storage_path,
            stop=stop_criteria,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=checkpoint_freq_iterations,
                num_to_keep=3,
                checkpoint_score_attribute=args.metric,
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
    """Parse CLI arguments and run training for the selected algorithm."""
    args = _parse_cli_args()
    configs = _load_configs(args)
    logger.info("Parsed arguments: %s", vars(args))

    slurm_resources = _init_ray(args.seed)

    exec_date_dt = datetime.datetime.now()
    exec_date = exec_date_dt.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algorithm}_seed{args.seed}_{exec_date}"
    storage_path = os.path.abspath(
        f"models/{configs.active_config.env_config_name}/ray/{args.algorithm}"
    )
    os.makedirs(storage_path, exist_ok=True)

    env_creator_config = {
        "env_config": configs.active_config,
        "data_combinator": configs.data_combinator,
        "reward_schedule_manager": configs.reward_manager,
    }
    register_env(
        "AdvBuilding",
        lambda cfg: adv_building_env_creator({**env_creator_config, **cfg}),
    )

    _, param_space = _build_algo_config(args, configs, slurm_resources, exec_date_dt)
    checkpoint_freq_iterations = _checkpoint_iterations(args, configs)
    tuner = _build_tuner(
        args, param_space, run_name, storage_path, checkpoint_freq_iterations,
    )

    experiment_path = os.path.join(storage_path, run_name)
    log_startup_banner(
        args=args,
        env_config=configs.active_config,
        training_param_config=configs.training_param_config,
        reward_manager=configs.reward_manager,
        data_combinator=configs.data_combinator,
        infra_combinator=configs.infra_combinator,
        slurm_resources=slurm_resources,
        run_name=run_name,
        experiment_path=experiment_path,
        storage_path=storage_path,
        seed=args.seed,
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

    _log_best_result(results, args.metric, storage_path)
    _dump_all_results(results)

    ray.shutdown()
    logger.info("Script completed")


if __name__ == "__main__":
    main()

# Usage examples:
# On slurm: sbatch slurm_scripts/slurm_train_ray.sh
#
# --load-config is REQUIRED — there is no default env config.
# python run_train_ray.py --algorithm ppo --load-config configs/env/env_test1_small.yaml --seed 42 --episodes 3500
# python run_train_ray.py --algorithm ppo --load-config configs/env/env_test1_mid.yaml --seed 42 --episodes 5000 --checkpoint-frequency-episodes 50 --metric achieved_reward
# python run_train_ray.py --algorithm sac --load-config configs/env/env_test1_large.yaml --seed 18 --episodes 3500 --metric reward_rate
