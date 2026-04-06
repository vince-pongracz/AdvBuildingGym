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
from pathlib import Path

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
from adv_building_gym import EnvConfigManager
from adv_building_gym.config import config as default_config, load_data_combinator_config
from adv_building_gym.config.reward_schedule_manager import RewardScheduleManager, RewardScheduleMode
from adv_building_gym.envs import adv_building_env_creator
from adv_building_gym.ray_training import common_model_setup, select_model
from adv_building_gym.config.training_param_config import TrainingParamConfig
from adv_building_gym.utils import (
    CustomJSONEncoder,
    RngService,
    SlurmResources,
    trial_dirname_creator,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # Override any existing logging configuration (e.g., from Ray/RLlib)
)
logger = logging.getLogger("main")

# Environment variables to control Ray/RLlib behavior (must be set before ray.init)
# These propagate to Ray worker processes
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::UserWarning"
# Suppress TensorFlow C++ logs (oneDNN, CUDA) and disable oneDNN custom ops
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Disable Ray metrics/event services (not needed for training, avoids connection errors in SLURM)
os.environ["RAY_METRICS_SERVICE_ENABLED"] = "0"
os.environ["RAY_event_stats"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
# os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
# Allow evaluation_interval > 1 without crashing on iterations that skip eval.
# Without this, tune.TuneConfig(metric="evaluation/env_runners/...") raises a
# ValueError when the metric key is absent from a non-eval iteration's results.
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

# Apply warning filters in the main process
setup_warning_filters()

# Build runtime environment variables to propagate to Ray workers
runtime_env_vars = {
    # Suppress deprecation/user warnings in worker processes
    # Note: comma-separated, not colon-separated
    "PYTHONWARNINGS": os.environ["PYTHONWARNINGS"],
    "TF_CPP_MIN_LOG_LEVEL": os.environ["TF_CPP_MIN_LOG_LEVEL"],
    "TF_ENABLE_ONEDNN_OPTS": os.environ["TF_ENABLE_ONEDNN_OPTS"],
    "RAY_METRICS_SERVICE_ENABLED": os.environ["RAY_METRICS_SERVICE_ENABLED"],
    "RAY_event_stats": os.environ["RAY_event_stats"],
    # Disable log deduplication (prevents "repeated Nx across cluster" messages)
    "RAY_DEDUP_LOGS": os.environ["RAY_DEDUP_LOGS"],
    # Disable ANSI color codes in non-interactive environments
    "RAY_COLOR_PREFIX": os.environ["RAY_COLOR_PREFIX"],
    "TERM": os.environ["TERM"],  # Prevents color output
}

logger.info("Runtime environment variables for Ray workers: %s", runtime_env_vars)


# Main
def main():
    """Parse CLI arguments and run training for selected algorithms."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", default="ppo", choices=["ppo", "sac", "ddpg", "td3", "a2c"]
    )
    parser.add_argument(
        "-cn", "--config_name", type=str,
        help="Name of the configuration file or experiment setup to use"
    )
    parser.add_argument(
        "--load-config", type=str,
        help="Path to YAML config file to load (e.g., 'configs/my_config.yaml')"
    )
    parser.add_argument(
        "--save-config", type=str,
        help="Path where to save the config as YAML (e.g., 'configs/my_config.yaml')"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Total training episodes. Primary stopping criterion. "
            "Converted to timesteps internally (episodes × EPISODE_LENGTH). "
            "Takes precedence over --timesteps if both are given."
    )
    parser.add_argument(
        "--timesteps", type=float, default=None,
        help="(Deprecated, prefer --episodes) Total environment timesteps. "
            "Ignored when --episodes is given. Legacy default: 1e6."
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments" # Change env number?
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--metric",
        type=str,
        default="reward_rate",
        choices=[
            # NOTE VP 2026.01.12: Diff between episode_return_mean and achieved_reward:
            # - achieved_reward: Custom metric from episode_callbacks.py
            #   Calculates sum(rewards) per episode, then averages across episodes in CURRENT iteration only (~14 episodes)
            #   More responsive to recent performance changes
            # - episode_return_mean: RLlib built-in metric
            #   Same base calculation (sum of rewards per episode), but uses exponential moving average (EMA)
            #   smoothed over last 25 episodes (metrics_num_episodes_for_smoothing=25)
            #   More stable, less sensitive to noise, better for detecting long-term trends
            # - reward_rate: Custom metric = achieved_reward / max_possible_reward
            #   Normalized performance score in [0, 1] range
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
        help="Save per-step trajectory JSON per episode (default: False)"
    )
    parser.add_argument(
        "--data-config", type=str, default=None,
        help="Path to data combinator YAML config (default: configs/data_scheduler/train_data_combinator_config.yaml)"
    )
    parser.add_argument(
        "--grad-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradual reward training (curriculum). "
            "When active, rewards are introduced according to the reward "
            "schedule config. When inactive, all rewards are active from start."
    )
    parser.add_argument(
        "--reward-schedule", type=str, default=None,
        help="Path to reward schedule YAML config "
            "(default: configs/reward_cfg/reward_schedule_train.yaml)"
    )
    parser.add_argument(
        "--infra-schedule", type=str, default=None,
        help="Path to infra schedule YAML config "
            "(e.g. 'configs/infra_schedule/infra_schedule_train.yaml'). "
            "When provided, infrastructure parameters cycle according "
            "to the schedule. When omitted, a single config is used."
    )

    # Load configs:
    # Load training hyperparameters (shared across select_model and checkpoint calc)
    training_param_config = TrainingParamConfig.from_yaml(
        Path(__file__).resolve().parent / "configs" / "training_param_config.yaml"
    )

    args = parser.parse_args()

    if args.seed is not None:
        training_param_config.seed = args.seed
    else:
        args.seed = training_param_config.seed

    # Load config from file if specified, otherwise use default
    if args.load_config:
        logger.info("Loading config from: %s", args.load_config)
        active_config = EnvConfigManager.load(args.load_config)
    else:
        active_config = default_config

    # Load data combinator from YAML (separate from env config)
    data_combinator = load_data_combinator_config(
        yaml_path=args.data_config,
        seed_override=args.seed,
    )

    # Load reward schedule config.
    # When --grad-train is active the manager uses its configured mode
    # (gradual_add / iterate / random).  Otherwise mode is forced to "off"
    # (all rewards active, no swapping).
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

    # Load infrastructure schedule config (optional).
    # When provided, infrastructure parameters cycle according to the schedule.
    infra_combinator = None
    if args.infra_schedule:
        from adv_building_gym.infra_combinator import InfraCombinator
        infra_combinator = InfraCombinator.from_yaml(args.infra_schedule)
        logger.info(
            "Infrastructure schedule enabled: mode=%s, %d configs, "
            "swap every %d iterations",
            infra_combinator.mode, len(infra_combinator.config_paths),
            infra_combinator.swap_every_n_iterations,
        )

    # Resolve stopping criterion: --episodes takes precedence over --timesteps.
    # Internally, RLlib always stops on num_env_steps_sampled_lifetime (timesteps),
    # so we convert episodes → timesteps here for a user-friendly interface.
    if args.episodes is not None:
        args.timesteps = args.episodes * active_config.EPISODE_LENGTH
        logger.info("Stopping after %d episodes (%d timesteps)", args.episodes, args.timesteps)
    elif args.timesteps is not None:
        args.timesteps = int(args.timesteps)
        args.episodes = args.timesteps // active_config.EPISODE_LENGTH
        logger.info("--timesteps is deprecated, prefer --episodes. "
                    "Stopping after %d timesteps (~%d episodes)", args.timesteps, args.episodes)
    else:
        args.episodes = training_param_config.max_episodes_to_run
        args.timesteps = args.episodes * active_config.EPISODE_LENGTH
        logger.info("Using default: %d episodes (%d timesteps)", args.episodes, args.timesteps)

    # Initialise the singleton component instances exactly once in the main process.
    # This triggers CSV parsing (e.g. EVState) here, and only here.
    # Ray worker subprocesses (EnvRunners, SAC actor, Learner) never call this —
    # they use the factory methods directly via adv_building_env_creator.
    active_config.init_singletons()

    # Action space is handled by env wrappers (FlattenAction + RescaleAction)
    # applied in env_creator.

    args.config_name = active_config.env_config_name if args.config_name is None else args.config_name

    # Save config to file if specified
    if args.save_config:
        logger.info("Saving config to: %s", args.save_config)
        EnvConfigManager.save(active_config, args.save_config)
        logger.info("Config saved successfully")

    # Add evaluation/env_runners/ prefix to metric if not already present
    # RLlib reports custom metrics under evaluation/env_runners/ in the results dict
    if not args.metric.startswith("evaluation/"):
        args.metric = f"evaluation/env_runners/{args.metric}"

    logger.info("Parsed arguments: %s", vars(args))
    # ------------------------------------------------
    # Configure Ray from SLURM / environment when available so Ray doesn't
    # attempt to acquire more resources than the job was allocated.
    # Prefer SLURM vars (SLURM_CPUS_PER_TASK, CUDA_VISIBLE_DEVICES) and fall
    # back to sensible defaults (2 cpus, 0 gpus).
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus = int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else 2

    # Only use GPUs that SLURM explicitly allocated via --gres=gpu.
    # SLURM sets CUDA_VISIBLE_DEVICES to the allocated GPU ids; if unset,
    # no GPU was booked and Ray must not try to acquire one.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        gpus = len([x for x in cuda_visible.split(",") if x.strip() != ""])
        if not torch.cuda.is_available():
            logger.error(
                "SLURM allocated GPUs (CUDA_VISIBLE_DEVICES=%s) but PyTorch "
                "cannot access CUDA. Check driver/CUDA toolkit setup.",
                cuda_visible,
            )
            logger.error("Exiting.")
            sys.exit(1)
    else:
        gpus = 0
        logger.error(
            "No GPU allocated (CUDA_VISIBLE_DEVICES is not set). "
            "Training requires a GPU — submit with --gres=gpu:1.",
        )
        logger.error("Exiting.")
        sys.exit(1)

    slurm_resources = SlurmResources(num_cpus=cpus, num_gpus=gpus)
    logger.info("Training on device: cuda (%d GPU(s) from SLURM)", slurm_resources.num_gpus)

    logger.info("Initializing Ray with cpus=%s gpus=%s (from SLURM/CUDA env)", slurm_resources.num_cpus, slurm_resources.num_gpus)
    ray.init(
        num_cpus=slurm_resources.num_cpus,
        num_gpus=slurm_resources.num_gpus,
        ignore_reinit_error=True,
        # Propagate warning suppression and color settings to all Ray workers
        runtime_env={"env_vars": runtime_env_vars},
        # Suppress Ray's internal logging noise
        logging_level=logging.INFO,
    )

    # Initialize centralized RNG service as a Ray Named Actor on the head node.
    # Must be called after ray.init() so the actor can be deployed.
    # All Ray workers discover this actor automatically via RngService.get().
    RngService.initialize(args.seed)

    # Create run name with timestamp (similar to SB3 naming convention)
    exec_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algorithm}_seed{args.seed}_{exec_date}"
    storage_path = os.path.abspath(f"models/{args.config_name}/ray/{args.algorithm}")
    os.makedirs(storage_path, exist_ok=True)

    env_creator_config = {
        "data_combinator": data_combinator,
        "reward_schedule_manager": reward_manager,
    }
    register_env("AdvBuilding", lambda cfg: adv_building_env_creator({**env_creator_config, **cfg}))

    # Build algorithm-specific config
    algo_config = select_model(
        algorithm=args.algorithm,
        episode_length=active_config.EPISODE_LENGTH,
        training_config=training_param_config,
    )

    # Apply common RLlib configuration (resource allocation, action space, and callbacks)
    algo_config = common_model_setup(
        config=algo_config,
        training_config=training_param_config,
        slurm_resources=slurm_resources,
        metrics_base_dir="ep_metrics",
        clip_actions=True,
        data_combinator=data_combinator,
        log_trajectories=args.log_trajectories,
        reward_schedule_manager=reward_manager,
        infra_combinator=infra_combinator,
    )

    # Convert the RLlib config into a Tune param space
    param_space = algo_config.to_dict()

    # Save parameter space for inspection -- 
    with open("param_space.json", "w", encoding="utf-8") as f:
        # NOTE: param_space stores both new and old API stuff for backward compatibility,
        # that is why the model dict contains the default values and _model_config the true specification
        json.dump(param_space, f, cls=CustomJSONEncoder, indent=4)

    # Convert episode-based checkpoint frequency to training iterations.
    # train_batch_size_per_learner drives how many timesteps RLlib processes
    # per training iteration — but its meaning differs by algorithm:
    #   PPO  — ppo_episodes_per_iteration × EPISODE_LENGTH (on-policy batch)
    #   SAC  — sac_replay_batch_size (off-policy replay buffer sample)
    timesteps_per_episode = active_config.EPISODE_LENGTH
    if args.algorithm == "ppo":
        timesteps_per_iteration = training_param_config.ppo_episodes_per_iteration * active_config.EPISODE_LENGTH
    else:
        timesteps_per_iteration = training_param_config.sac_replay_batch_size
    checkpoint_freq_iterations = max(1, int(
        (args.checkpoint_frequency_episodes * timesteps_per_episode) / timesteps_per_iteration
    ))

    logger.info(
        "Checkpoint configuration: every %d iterations (~%d episodes), metric=%s",
        checkpoint_freq_iterations,
        args.checkpoint_frequency_episodes,
        args.metric,
    )

    # Setup stopping criteria and run configuration for the tuner
    # Note: In the new API stack, use 'num_env_steps_sampled_lifetime' instead of 'timesteps_total'
    # Episodes are converted to timesteps above (--episodes × EPISODE_LENGTH)
    stop_criteria = {
        "num_env_steps_sampled_lifetime": args.timesteps,
    }

    # Configure progress reporter to show training metrics
    progress_reporter = CLIReporter(
        metric_columns={
            "training_iteration": "Iter",
            "num_env_steps_sampled_lifetime": "Steps",
            args.metric: "Metric",
            "evaluation/env_runners/episode_return_mean": "EpRet",
            "evaluation/env_runners/achieved_reward": "AchRew",
            "evaluation/env_runners/reward_rate": "RewRate",
        },
        max_report_frequency=30,  # Report every 30 seconds
        print_intermediate_tables=True,
    )

    tuner = tune.Tuner(
        args.algorithm.upper(),  # e.g.: "PPO" or "SAC"
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
            # TODO VP 2026.03.20. : Read more about TuneConfig params -- needed when tune hyperparam optimisation is used...
            # search_alg=,
            # scheduler=
        ),
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=storage_path,
            stop=stop_criteria,
            # Ray Tune handles both periodic and best-model checkpointing.
            # checkpoint_score_attribute selects the best checkpoint by metric.
            # Link: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=checkpoint_freq_iterations,
                num_to_keep=3,
                checkpoint_score_attribute=args.metric,
                checkpoint_score_order="max",
            ),
            progress_reporter=progress_reporter,
            verbose=2,  # 0=silent, 1=less, 2=default, 3=verbose
        ),
    )

    experiment_path = os.path.join(storage_path, run_name)
    logger.info("Starting tuner.fit() for: %s", run_name)
    logger.info("=" * 70)
    logger.info(
        "To visualize results with TensorBoard, run:\n"
        "  tensorboard --logdir %s", experiment_path
    )
    logger.info("=" * 70)
    logger.info("Training progress will be displayed below:")
    logger.info("=" * 70)

    t0 = time.time()
    results = tuner.fit()

    elapsed_time = (time.time() - t0) / 60
    logger.info("=" * 70)
    logger.info("Tuner/training finished in %.2f min", elapsed_time)
    logger.info("=" * 70)

    # ===================================================================================
    # ACCESS training results and checkpoint locations
    # Get best result with safe metric access
    try:
        best_result = results.get_best_result(
            metric=args.metric,
            mode="max"
        )

        # Log checkpoint and trial paths
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

        # Safely access nested metrics
        if best_result and best_result.metrics:
            env_runners_metrics = best_result.metrics.get("env_runners", {})

            # Extract key metrics
            metrics_to_log = {
                "episode_return_mean": env_runners_metrics.get("episode_return_mean"),
                "achieved_reward": env_runners_metrics.get("achieved_reward"),
                "reward_rate": env_runners_metrics.get("reward_rate"),
            }

            logger.info("Best performing trial's final reported metrics:")
            # Log the optimization metric first
            metric_key = args.metric.split("/")[-1]  # e.g., "reward_rate"
            opt_value = metrics_to_log.get(metric_key)
            if opt_value is not None:
                logger.info("Optimized metric (%s): %.4f", args.metric, opt_value)

            # Log all available metrics
            for name, value in metrics_to_log.items():
                if value is not None and name != metric_key:
                    logger.info("  %s: %.4f", name, value)
        else:
            logger.warning("No best result or metrics available")

    except Exception as e:
        logger.error("Error retrieving best result: %s", str(e))
    # ===================================================================================


    # TODO VP 2026.03.20. : Clean this up, refactor
    for i, res in enumerate(results._results):
        with open(f"result_{i}.json", "w", encoding="utf-8") as f:
            # Handle failed trials where config/metrics may be None
            config_data = {}
            if res.config is not None:
                try:
                    config_data = res.config.copy() if hasattr(res.config, 'copy') else dict(res.config)
                    # Remove non-JSON-serializable objects from config
                    if "_rl_module_spec" in config_data:
                        del config_data["_rl_module_spec"]
                except (TypeError, ValueError) as e:
                    logger.warning("Could not serialize config for result %d: %s", i, e)
                    config_data = {"error": str(e)}

            dump = {
                "config": config_data,
                "error": getattr(res, 'error', None),
                "metrics": res.metrics if res.metrics is not None else {},
                "path": res.path,
            }
            json.dump(dump, f, cls=CustomJSONEncoder, indent=4)


    # Ensure ray resources are released before exiting
    ray.shutdown()
    logger.info("Script completed")


if __name__ == "__main__":
    main()

# Usage examples:
# On slurm: sbatch slurm_scripts/slurm_train_ray.sh

# Default settings (3500 episodes, checkpoint every 20 episodes, optimize reward_rate)
# python run_train_ray.py --algorithm ppo --seed 42 --episodes 3500

# Custom episode count, checkpoint frequency, and metric
# python run_train_ray.py --algorithm ppo --seed 42 --episodes 5000 --checkpoint-frequency-episodes 50 --metric achieved_reward

# SAC with specific config name
# python run_train_ray.py --algorithm sac --seed 18 -cn env_test1_{s/m/l} --episodes 3500 --metric reward_rate

