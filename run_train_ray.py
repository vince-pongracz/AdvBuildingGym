

# TODO VP 2026.01.13. : add docstring to the file

import os
import time
import datetime
import logging

import json
import argparse
import torch

import ray
from ray import tune
from ray.tune import CLIReporter

# Import warning filter setup from utils
# The function is centrally defined in adv_building_gym/utils/warning_filters.py
# and is called in two places:
#   1. Here in the main process (before Ray starts)
#   2. In AdvBuildingGym.__init__() (runs in each Ray worker when env is created)
from adv_building_gym.utils import setup_warning_filters

# Trigger registration of the custom Gym IDs
from adv_building_gym import make_checkpoint_callback_class
from adv_building_gym.env_config import config as env_config
from adv_building_gym.envs import adv_building_env_creator
from adv_building_gym.ray_training import common_model_config, select_model
from adv_building_gym.utils import (
    CustomJSONEncoder,
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
# Disable Ray metrics/event services (not needed for training, avoids connection errors in SLURM)
os.environ["RAY_METRICS_SERVICE_ENABLED"] = "0"
os.environ["RAY_event_stats"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
# os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

# Apply warning filters in the main process
setup_warning_filters()

# Build runtime environment variables to propagate to Ray workers
runtime_env_vars = {
    # Suppress deprecation/user warnings in worker processes
    # Note: comma-separated, not colon-separated
    "PYTHONWARNINGS": os.environ["PYTHONWARNINGS"],
    "RAY_METRICS_SERVICE_ENABLED": os.environ["RAY_METRICS_SERVICE_ENABLED"],
    "RAY_event_stats": os.environ["RAY_event_stats"],
    # Disable log deduplication (prevents "repeated Nx across cluster" messages)
    "RAY_DEDUP_LOGS": os.environ["RAY_DEDUP_LOGS"],
    # Disable ANSI color codes in non-interactive environments
    "RAY_COLOR_PREFIX": os.environ["RAY_COLOR_PREFIX"],
    "TERM": os.environ["TERM"],  # Prevents color output
}

logger.info("Runtime environment variables for Ray workers: %s", runtime_env_vars)


ENV_ID: str = "AdvBuilding"


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
    parser.add_argument("--timesteps", type=float, default=1e6) # a million
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments" # Change env number?
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument(
        "--training",
        action="store_true",
        help="Use training split of price data (else test split)",
    )
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
            #   smoothed over last 100 episodes (metrics_num_episodes_for_smoothing=100)
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
    args = parser.parse_args()

    args.timesteps = int(args.timesteps)
    args.config_name = env_config.config_name if args.config_name is None else args.config_name
    # Add env_runners/ prefix to metric if not already present
    if not args.metric.startswith("env_runners/"):
        args.metric = f"env_runners/{args.metric}"
    
    logger.info("Parsed arguments: %s", vars(args))
    # ------------------------------------------------
    # Configure Ray from SLURM / environment when available so Ray doesn't
    # attempt to acquire more resources than the job was allocated.
    # Prefer SLURM vars (SLURM_CPUS_PER_TASK, CUDA_VISIBLE_DEVICES) and fall
    # back to sensible defaults (2 cpus, 0 gpus).
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus = int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        # CUDA_VISIBLE_DEVICES may contain comma-separated GPU ids
        gpus = len([x for x in cuda_visible.split(",") if x.strip() != ""])
    else:
        gpus = 0

    logger.info("Initializing Ray with cpus=%s gpus=%s (from SLURM/CUDA env)", cpus, gpus)

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        ignore_reinit_error=True,
        # Propagate warning suppression and color settings to all Ray workers
        runtime_env={"env_vars": runtime_env_vars},
        # Suppress Ray's internal logging noise
        logging_level=logging.INFO,
    )

    # Create run name with timestamp (similar to SB3 naming convention)
    exec_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algorithm}_seed{args.seed}_{exec_date}"
    storage_path = os.path.abspath(f"models/{args.config_name}/ray/{args.algorithm}")
    os.makedirs(storage_path, exist_ok=True)

    # Checkpoint directory path for callback-level checkpointing (similar to SB3)
    # Note: Directory will be created lazily by BestModelCheckpointCallback when first checkpoint is saved
    checkpoint_dir = os.path.abspath(f"{storage_path}/checkpoints_{run_name}")

    # Create checkpoint callback class for best model tracking -- easier to hand it over already instantiated, like this
    checkpoint_callback_class = make_checkpoint_callback_class(
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency_episodes,
        num_to_keep=1,  # Only keep best checkpoint at callback level
        metric=args.metric,
    )

    # Build algorithm-specific config
    algo_config = select_model(
        algorithm=args.algorithm,
        episode_length=env_config.EPISODE_LENGTH,
    )

    # Apply common RLlib configuration (resource allocation, action space, and callbacks)
    algo_config = common_model_config(
        config=algo_config,
        seed=args.seed,
        env_creator=adv_building_env_creator,
        num_cpus=cpus,
        num_gpus=gpus,
        checkpoint_callback_class=checkpoint_callback_class,
        env_id=ENV_ID,
        rewards=env_config.rewards,
        metrics_base_dir="ep_metrics",
        clip_actions=True,
    )

    # Convert the RLlib config into a Tune param space
    param_space = algo_config.to_dict()

    # Save parameter space for inspection -- 
    with open("param_space.json", "w", encoding="utf-8") as f:
        # NOTE: param_space stores both new and old API stuff for backward compatibility, 
        # that is why the model dict contains the default values and _model_config the true specification
        json.dump(param_space, f, cls=CustomJSONEncoder, indent=4)

    # Calculate checkpoint frequency in training iterations based on episodes
    # Episode length from env config (288 timesteps per episode)
    # Training batch size per iteration: train_batch_size_per_learner = 4000 timesteps
    timesteps_per_episode = env_config.EPISODE_LENGTH  # 288 timesteps
    timesteps_per_iteration = param_space.get("train_batch_size_per_learner", 4000)
    checkpoint_freq_iterations = max(1, int((args.checkpoint_frequency_episodes * timesteps_per_episode) / timesteps_per_iteration))

    logger.info("Checkpoint configuration:")
    logger.info("  Callback-level (best model): every %d episodes, metric=%s, dir=%s",
        args.checkpoint_frequency_episodes,
        args.metric,
        checkpoint_dir
    )
    logger.info("  Ray Tune-level: every %d iterations (%.1f timesteps/episode, %d timesteps/iteration), num_to_keep=5",
        checkpoint_freq_iterations,
        timesteps_per_episode,
        timesteps_per_iteration
    )

    # Setup stopping criteria and run configuration for the tuner
    # Note: In the new API stack, use 'num_env_steps_sampled_lifetime' instead of 'timesteps_total'
    stop_criteria = {
        "num_env_steps_sampled_lifetime": args.timesteps,
        "training_iteration": 250,
    }

    # Configure progress reporter to show training metrics
    progress_reporter = CLIReporter(
        metric_columns={
            "training_iteration": "Iter",
            "num_env_steps_sampled_lifetime": "Steps",
            args.metric: "Metric",
            "env_runners/episode_return_mean": "EpRet",
            "env_runners/achieved_reward": "AchRew",
            "env_runners/reward_rate": "RewRate",
        },
        max_report_frequency=30,  # Report every 30 seconds
        print_intermediate_tables=True,
    )

    tuner = tune.Tuner(
        args.algorithm.upper(),  # "PPO"
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
                checkpoint_frequency=checkpoint_freq_iterations,  # Based on episode frequency
                num_to_keep=5,  # Keep 5 checkpoints to reduce experiment state snapshot frequency
                # Increasing num_to_keep reduces Ray Tune's experiment state snapshotting overhead
                # (see TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S warning)
                # Best model tracking is handled by BestModelCheckpointCallback at callback level
                checkpoint_score_attribute=args.metric,
                checkpoint_score_order="max",
                ),
            progress_reporter=progress_reporter,
            verbose=2,  # 0=silent, 1=less, 2=default, 3=verbose
        ),
    )

    logger.info("Starting tuner.fit() for: %s", run_name)
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

            # Get checkpoint information (Ray Tune-level)
            if best_result.checkpoint:
                checkpoint_path = best_result.checkpoint.path
                logger.info("  Ray Tune checkpoint: %s", checkpoint_path)
            else:
                logger.warning("  No Ray Tune checkpoint available for best trial")

            # Callback-level checkpoint (best model only)
            logger.info("  Callback checkpoint (best model): %s", checkpoint_dir)
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

# Default settings (checkpoint every 20 episodes, optimize reward_rate)
# python run_train_ray.py --algorithm ppo --seed 42 --timesteps 1e6

# Custom checkpoint frequency and metric
# python run_train_ray.py --algorithm ppo --seed 42 --timesteps 1e6 --checkpoint-frequency-episodes 50 --metric achieved_reward

# With specific config name
# python run_train_ray.py -a sac -s 18 -cn test1 --checkpoint-frequency-episodes 30 --metric reward_rate
