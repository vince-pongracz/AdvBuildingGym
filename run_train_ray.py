

"""
run_train_rl.py – Train PPO, SAC, DDPG, TD3, or A2C on LLEC-HeatPumpHouse environments.
This script sets up parallel training and evaluation environments with configurable observation
and reward settings. It trains an RL agent using Stable Baselines3 and saves the model and logs.
"""

import os
import sys
import time
import datetime
import logging

import numpy as np
import json
import argparse
import torch
import gymnasium as gym

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.connectors.env_to_module import FlattenObservations

# Import warning filter setup from utils
# The function is centrally defined in adv_building_gym/utils/warning_filters.py
# and is called in two places:
#   1. Here in the main process (before Ray starts)
#   2. In AdvBuildingGym.__init__() (runs in each Ray worker when env is created)
from adv_building_gym.utils import setup_warning_filters

# Apply warning filters in the main process
setup_warning_filters()

# Environment variables to control Ray/RLlib behavior (must be set before ray.init)
# These propagate to Ray worker processes
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::UserWarning"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
# Disable Ray metrics/event services (not needed for training, avoids connection errors in SLURM)
# os.environ["RAY_METRICS_SERVICE_ENABLED"] = "0"
# os.environ["RAY_event_stats"] = "0"
# os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

# Trigger registration of the custom Gym IDs
from adv_building_gym import AdvBuildingGym, create_on_episode_end_callback, make_checkpoint_callback_class
from adv_building_gym.env_config import config as env_config
from adv_building_gym.utils import (
    CustomJSONEncoder,
    trial_dirname_creator,
    ResourceAllocation,
    validate_resource_allocation,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # Override any existing logging configuration (e.g., from Ray/RLlib)
)
logger = logging.getLogger("main")

ENV_ID: str = "AdvBuilding"

# Create episode end callback using the factory function
on_episode_end_cb = create_on_episode_end_callback(
    env_id=ENV_ID,
    rewards=env_config.rewards,
    metrics_base_dir="ep_metrics",
    exec_date=datetime.datetime.now()
)


def select_model(
    algorithm: str,
    seed: int,
    timesteps: int,
    checkpoint_dir: str,
    checkpoint_frequency: int,
    metric: str,
    num_cpus: int,
    num_gpus: int,
    clip_actions: bool = True,
):
    """
    Selects and configures a model for training based on the algorithm name.

    Args:
        algorithm: RL algorithm to use (e.g., "ppo")
        seed: Random seed for reproducibility
        timesteps: Total training timesteps
        checkpoint_dir: Directory for callback-level checkpoints
        checkpoint_frequency: Checkpoint every N episodes
        metric: Metric to optimize for checkpointing
        num_cpus: Total CPUs available (from Ray/SLURM)
        num_gpus: Total GPUs available (from Ray/SLURM)
        clip_actions: Whether to clip actions to action space bounds
    """
    
    # Hyperparameters for consistent evaluation across algorithms
    learning_rate = 3e-4
    batch_size = 64
    learning_starts = 10 * env_config.EPISODE_LENGTH
    n_steps = env_config.EPISODE_LENGTH

    # Resource allocation:
    # - Learners: one per GPU, each gets 1 GPU and 1 CPU
    # - Driver: 1 CPU (taken from env_runners pool)
    # - Env runners: remaining CPUs after learners and driver
    num_learners = max(1, num_gpus)  # At least 1 learner even without GPU
    num_gpus_per_learner = 1 if num_gpus > 0 else 0
    num_cpus_per_learner = 1
    num_cpus_per_env_runner = 1

    driver_cpus = 1
    learner_total_cpus = num_learners * num_cpus_per_learner
    remaining_cpus = num_cpus - learner_total_cpus - driver_cpus
    num_env_runners = max(1, remaining_cpus // num_cpus_per_env_runner)

    logger.info(
        "Resource allocation in select_model: learners=%d (gpus=%d, cpus=%d each), "
        "env_runners=%d (cpus=%d each), driver=%d CPU",
        num_learners, num_gpus_per_learner, num_cpus_per_learner,
        num_env_runners, num_cpus_per_env_runner, driver_cpus
    )

    # Get observation and action spaces from a temporary env instance
    # NOTE: We only provide action_space explicitly. The observation_space is
    # intentionally NOT provided because FlattenObservations connector transforms
    # the Dict obs space into a flat Box. If we provide the Dict space here,
    # RLlib's Catalog tries to build an encoder for Dict (unsupported) before
    # the connector can transform it.
    _obs_space, action_space = get_env_spaces()
    logger.debug("Env action_space: %s (obs_space inferred after FlattenObservations)", action_space)

    if algorithm == "ppo":
        config = PPOConfig()
        # NOTE: These values should match SB3 PPO hyperparameters for fair comparison:
        #   - train_batch_size_per_learner: Total timesteps collected before training update
        #     SB3 equivalent: n_steps * num_envs = 288 * 4 = 1,152
        #     We use 4000 here to account for 2 env_runners collecting in parallel
        #   - minibatch_size: SGD minibatch size for gradient updates (like SB3's batch_size=64)
        #   - num_epochs: Number of passes over collected data (SB3 default is 10, we use 4)
        config.training(
            lr=learning_rate,
            # TODO VP 2026.01.07. : Use timesteps param?
            train_batch_size_per_learner=4000,  # ~14 episodes worth of data before each training update
            minibatch_size=batch_size,  # 64 - SGD minibatch size (same as SB3)
            num_epochs=4,  # Number of epochs per training iteration (typical for PPO)
        )

    # TODO VP 2025.12.11. : add more RL algos
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    config = config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    config.environment(
        env="AdvBuilding",
        # observation_space intentionally omitted - inferred after FlattenObservations
        action_space=action_space,
        normalize_actions=True,
        clip_actions=clip_actions,
    )
    config.debugging(
        # WARN: Reduces verbosity (suppress connector pipeline INFO messages)
        log_level="INFO",
        log_sys_usage=True,
        seed=seed
    )
    config.reporting(
        keep_per_episode_custom_metrics=True,
    )
    config.framework(
        framework="torch",
        eager_tracing=True,
        eager_max_retraces=20,
        tf_session_args={},
        local_tf_session_args={},
    )
    # NOTE VP 2026.01.08. : about ray and rllib concept https://docs.ray.io/en/latest/rllib/key-concepts.html
    # Learning the NN, policy (gradient updates) -- needs GPU
    config.learners(
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        num_cpus_per_learner=num_cpus_per_learner,
    )
    # Sampling actions (querying the env, using the policy, sample trajectories) -- no GPU needed
    config.env_runners(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        num_gpus_per_env_runner=0,
        # Flatten dict observation space into a single vector for the RL module
        # NOTE VP 2026.01.05. : if other observation space needed for the policy, change the observations in the env.
        # (rather than using a new observation encoder -- that must be learnt as well, it overcomplicates things...)
        env_to_module_connector=lambda env, spaces, device: FlattenObservations(),  # type: ignore
    )
    config.rl_module(
        # Use new API to avoid RLModule(config=RLModuleConfig) deprecation warning
        model_config=DefaultModelConfig(
            fcnet_activation='relu',
            fcnet_hiddens=[32, 32, 32],
        ),
    )
    config.evaluation(
        evaluation_interval=1,  # run evaluation every train() iteration
        evaluation_duration_unit="episodes",
        evaluation_duration=2,  # e.g., 2 episodes
        # True only if `evaluation_num_env_runners` > 0
        evaluation_parallel_to_training=False,
    )

    config.logger_config = {
        "type": "ray.tune.logger.UnifiedLogger",
        "loggers": [
                "ray.tune.json.JsonLoggerCallback",
                "ray.tune.csv.CSVLoggerCallback",
                "ray.tune.tensorboardx.TBXLoggerCallback",
        ],
    }

    # Configure callbacks:
    # - on_episode_end: Logs episode metrics (reward_rate, achieved_reward, etc.)
    # - Checkpoint callback: Saves best model based on metric every N episodes
    #
    # RLlib calls on_episode_end callbacks first, then callback class methods -- ensures metrics are logged before checkpoint decisions are made.
    config.callbacks(
        make_checkpoint_callback_class(
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            num_to_keep=1,  # Only keep best checkpoint at callback level
            metric=metric,
        ),
        on_episode_end=on_episode_end_cb,  # Episode metrics callback (runs first)
    )

    return config


def env_creator(config):
    """Factory function for Ray Tune to create AdvBuildingGym instances."""
    return AdvBuildingGym(
        infras=env_config.infras,
        datasources=env_config.datasources,
        rewards=env_config.rewards,
        building_props=env_config.building_props,
    )


def get_env_spaces():
    """
    Instantiate env temporarily to extract observation and action spaces.

    This avoids RLlib inferring spaces from remote workers, making the
    configuration explicit and reducing startup overhead.
    """
    temp_env = env_creator({})
    obs_space = temp_env.observation_space
    action_space = temp_env.action_space
    temp_env.close()
    return obs_space, action_space


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

    # TODO VP 2026.01.06. : Get these variables from the SLURM env?
    # Build runtime environment variables to propagate to Ray workers
    runtime_env_vars = {
        # Suppress deprecation/user warnings in worker processes
        # Note: comma-separated, not colon-separated
        "PYTHONWARNINGS": "ignore::DeprecationWarning,ignore::UserWarning",
        "RAY_METRICS_SERVICE_ENABLED": "0",
        "RAY_event_stats": "0",
        # Disable log deduplication (prevents "repeated Nx across cluster" messages)
        "RAY_DEDUP_LOGS": "0",
        # Disable ANSI color codes in non-interactive environments
        "RAY_COLOR_PREFIX": "0",
        "TERM": "dumb",  # Prevents color output
    }

    # Override RAY_COLOR_PREFIX from environment if explicitly set
    if "RAY_COLOR_PREFIX" in os.environ:
        runtime_env_vars["RAY_COLOR_PREFIX"] = os.environ["RAY_COLOR_PREFIX"]
        logger.info("Overriding RAY_COLOR_PREFIX=%s from environment", os.environ["RAY_COLOR_PREFIX"])

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        ignore_reinit_error=True,
        # Propagate warning suppression and color settings to all Ray workers
        runtime_env={"env_vars": runtime_env_vars},
        # Suppress Ray's internal logging noise
        logging_level=logging.INFO,
    )
    
    # Map usecase to environment Gym Environment ID alias per reward mode (registered by adv_building_gym)
    tune.register_env(ENV_ID, env_creator)

    # Create run name with timestamp (similar to SB3 naming convention)
    exec_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algorithm}_seed{args.seed}_{exec_date}"
    storage_path = os.path.abspath(f"models/{args.config_name}/ray/{args.algorithm}")
    os.makedirs(storage_path, exist_ok=True)

    # Checkpoint directory path for callback-level checkpointing (similar to SB3)
    # Note: Directory will be created lazily by BestModelCheckpointCallback when first checkpoint is saved
    checkpoint_dir = os.path.abspath(f"{storage_path}/checkpoints_{run_name}")

    # Build algorithm config with checkpoint parameters and resource allocation
    algo_config = select_model(
        algorithm=args.algorithm,
        seed=args.seed,
        timesteps=args.timesteps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency_episodes,
        metric=args.metric,
        num_cpus=cpus,
        num_gpus=gpus,
    )
    # Convert the RLlib config into a Tune param space
    param_space = algo_config.to_dict()

    # Extract resource allocation from config (set by select_model)
    num_learners = int(param_space.get("num_learners", 1))
    cpus_per_learner = int(param_space.get("num_cpus_per_learner", 1))
    cpus_per_env_runner = int(param_space.get("num_cpus_per_env_runner", 1))
    actual_env_runners = int(param_space.get("num_env_runners", 1))

    # Calculate total CPU usage for validation
    driver_cpus = 1
    learner_total_cpus = num_learners * cpus_per_learner
    total_cpu_usage = driver_cpus + learner_total_cpus + (actual_env_runners * cpus_per_env_runner)
    unused_cpus = cpus - total_cpu_usage

    # Validate resource allocation against SLURM constraints
    allocation = ResourceAllocation(
        total_cpu_usage=total_cpu_usage,
        unused_cpus=unused_cpus,
        driver_cpus=driver_cpus,
        num_learners=num_learners,
        cpus_per_learner=cpus_per_learner,
        learner_total_cpus=learner_total_cpus,
        actual_env_runners=actual_env_runners,
        cpus_per_env_runner=cpus_per_env_runner,
        slurm_cpus=cpus,
        slurm_gpus=gpus,
    )
    validate_resource_allocation(allocation, param_space)

    with open("param_space.json", "w", encoding="utf-8") as f:
        dump = dict(param_space.copy())
        json.dump(dump, f, cls=CustomJSONEncoder, indent=4)

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
