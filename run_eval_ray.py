"""
run_eval_ray.py – Evaluate Ray/RLlib trained models on AdvBuildingGym.

This script loads a trained Ray/RLlib checkpoint and evaluates it on the environment
in inference mode, logging episode metrics and performance statistics.
"""


import os
import sys
import argparse
import time
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

import ray
from ray.rllib.algorithms import Algorithm

from adv_building_gym import AdvBuildingGym, ConfigManager
from adv_building_gym.config import config as default_config
from adv_building_gym.utils import CustomJSONEncoder, TrajectoryCollector, setup_warning_filters

# Apply warning filters
setup_warning_filters()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True
)
logger = logging.getLogger("main")


def find_best_checkpoint(base_path: str) -> str:
    """
    Find the best-performing Ray checkpoint by reading best_checkpoint_metadata.json
    files saved by BestModelCheckpointCallback during training.

    Searches for metadata files in checkpoint directories under base_path,
    selects the one with the highest metric value.

    Args:
        base_path: Root directory to search (e.g., models/{config}/ray/{algo})

    Returns:
        str: Path to the best checkpoint directory

    Raises:
        FileNotFoundError: If no checkpoint metadata is found
    """
    candidates = []

    for root, dirs, files in os.walk(base_path):
        if "best_checkpoint_metadata.json" in files:
            metadata_path = os.path.join(root, "best_checkpoint_metadata.json")
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metric_value = metadata.get("best_metric_value", -np.inf)
                checkpoint_path = metadata.get("checkpoint_path", "")
                if checkpoint_path and os.path.exists(checkpoint_path):
                    candidates.append((metric_value, checkpoint_path, metadata))
                    logger.info(
                        "  Found checkpoint: %s=%s, path=%s",
                        metadata.get("metric", "unknown"),
                        metric_value,
                        checkpoint_path,
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read metadata %s: %s", metadata_path, e)

    if not candidates:
        raise FileNotFoundError(
            f"No best_checkpoint_metadata.json found in {base_path}"
        )

    # Sort by metric value descending, pick best
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_value, best_path, best_metadata = candidates[0]

    logger.info(
        "Selected best checkpoint: %s=%.4f, episode=%d, path=%s",
        best_metadata.get("metric", "unknown"),
        best_value,
        best_metadata.get("episode", -1),
        best_path,
    )
    return best_path


def find_latest_checkpoint(base_path: str = "models") -> str:
    """
    Fallback: find the most recent Ray checkpoint in the models directory
    by modification time. Used when no best_checkpoint_metadata.json exists.

    Returns:
        str: Path to the checkpoint directory
    """
    checkpoint_paths = []

    for root, dirs, files in os.walk(base_path):
        # Ray checkpoints contain either .pkl files (older) or
        # algorithm_state.pkl / .is_checkpoint marker files (newer)
        is_checkpoint = (
            "checkpoint_" in root
            and any(
                f.endswith(".pkl") or f == ".is_checkpoint"
                for f in files
            )
        ) or (
            # Callback-saved best_model checkpoints
            "best_model_" in os.path.basename(root)
            and any(f.endswith(".pkl") or f == ".is_checkpoint" for f in files)
        )

        if is_checkpoint:
            mtime = os.path.getmtime(root)
            checkpoint_paths.append((mtime, root))

    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {base_path}")

    # Sort by modification time and return most recent
    checkpoint_paths.sort(reverse=True)
    latest_checkpoint = checkpoint_paths[0][1]

    logger.info(
        "Found %d checkpoints, using latest: %s",
        len(checkpoint_paths),
        latest_checkpoint,
    )
    return latest_checkpoint


def evaluate_ray_model(
    checkpoint_path: str,
    active_config,
    num_episodes: int = 1,
    seed: int = 42,
    save_results: bool = True,
    output_dir: str = "eval_results",
    log_trajectories: bool = True,
):
    """
    Evaluate a Ray/RLlib trained model on AdvBuildingGym.

    Args:
        checkpoint_path: Path to the Ray checkpoint directory
        active_config: Config object with infras, statesources, rewards, building_props
        num_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        save_results: Whether to save results to file
        output_dir: Directory to save evaluation results
        log_trajectories: Whether to save per-step trajectory JSON per episode

    Returns:
        dict: Evaluation statistics
    """
    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = 1 if torch.cuda.is_available() else 0

    logger.info("=" * 70)
    logger.info("Starting Ray model evaluation")
    logger.info("  Checkpoint: %s", checkpoint_path)
    logger.info("  Config: %s", active_config.config_name)
    logger.info("  Episodes: %d", num_episodes)
    logger.info("  Seed: %d", seed)
    logger.info("  Device: %s", device)
    logger.info("  GPUs available: %d", num_gpus)
    logger.info("=" * 70)

    # Initialize Ray with GPU support if available
    if not ray.is_initialized():
        ray.init(
            num_cpus=2,
            num_gpus=num_gpus,
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
        )

    # Load the algorithm from checkpoint
    logger.info("Loading algorithm from checkpoint...")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # Get algorithm config for logging
    algo_config = algo.config
    algorithm_name = algo_config.get("framework", "unknown")
    logger.info("Algorithm: %s", algo.__class__.__name__)
    logger.info("Framework: %s", algorithm_name)

    # Create evaluation environment using the active config (matches training)
    logger.info("Creating evaluation environment...")
    env = AdvBuildingGym(
        infras=active_config.infras,
        statesources=active_config.statesources,
        rewards=active_config.rewards,
        building_props=active_config.building_props,
        training=False,  # Evaluation mode
    )

    # Enable full state logging in info dicts for trajectory collection
    if log_trajectories:
        env.log_full_info = True

    # Set up trajectory collector
    collector = TrajectoryCollector(env) if log_trajectories else None

    max_reward_per_step = sum(r.weight * r.max_reward for r in active_config.rewards)

    # Evaluation loop
    episode_stats = []
    all_rewards = []
    start_time = time.time()

    for ep in range(num_episodes):
        logger.info("=" * 50)
        logger.info("Episode %d/%d", ep + 1, num_episodes)

        obs, reset_info = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_rewards = []

        if collector is not None:
            collector.reset()
            collector.on_reset(reset_info)

        while not done:
            # Compute action using the policy (inference mode)
            raw_action = algo.compute_single_action(obs, explore=False)

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(raw_action)
            done = terminated or truncated

            if collector is not None:
                collector.on_step(
                    step=episode_length,
                    obs=obs,
                    action=raw_action,
                    reward=reward,
                    info=step_info,
                    raw_policy_action=raw_action,
                )

            episode_reward += reward
            episode_length += 1
            episode_rewards.append(reward)

            obs = next_obs

        # Calculate episode statistics (consistent with episode_callbacks.py)
        achieved_reward = np.sum(episode_rewards)
        max_achievable_reward = episode_length * max_reward_per_step
        reward_rate = achieved_reward / max_achievable_reward if max_achievable_reward > 0 else 0.0

        ep_stats = {
            "episode": ep + 1,
            "length": episode_length,
            "total_reward": float(episode_reward),
            "achieved_reward": float(achieved_reward),
            "max_achievable_reward": float(max_achievable_reward),
            "reward_rate": float(reward_rate),
            "seed": seed + ep,
        }

        # Save per-episode trajectory JSON
        if collector is not None and save_results:
            collector.on_episode_end(
                episode_id=ep,
                seed=seed + ep,
                metadata={
                    "config_name": active_config.config_name,
                    "checkpoint_path": checkpoint_path,
                    "algorithm": algo.__class__.__name__,
                },
            )
            traj_file = os.path.join(output_dir, f"{ep}_trajectory.json")
            collector.save_json(traj_file)

        episode_stats.append(ep_stats)
        all_rewards.append(episode_reward)

        logger.info("  Length: %d", episode_length)
        logger.info("  Total Reward: %.2f", episode_reward)
        logger.info("  Achieved Reward: %.2f", achieved_reward)
        logger.info("  Max Achievable: %.2f", max_achievable_reward)
        logger.info("  Reward Rate: %.4f", reward_rate)

    eval_time = time.time() - start_time

    # Compute summary statistics
    summary_stats = {
        "checkpoint_path": checkpoint_path,
        "config_name": active_config.config_name,
        "algorithm": algo.__class__.__name__,
        "num_episodes": num_episodes,
        "seed": seed,
        "eval_time_seconds": eval_time,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
        "mean_reward_rate": float(np.mean([s["reward_rate"] for s in episode_stats])),
        "std_reward_rate": float(np.std([s["reward_rate"] for s in episode_stats])),
        "episodes": episode_stats,
    }

    # Log summary
    logger.info("=" * 70)
    logger.info("Evaluation Summary")
    logger.info("  Mean Reward: %.2f +/- %.2f", summary_stats["mean_reward"], summary_stats["std_reward"])
    logger.info("  Mean Reward Rate: %.4f +/- %.4f", summary_stats["mean_reward_rate"], summary_stats["std_reward_rate"])
    logger.info("  Min/Max Reward: %.2f / %.2f", summary_stats["min_reward"], summary_stats["max_reward"])
    logger.info("  Evaluation time: %.2f seconds", eval_time)
    logger.info("=" * 70)

    # Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename based on checkpoint directory name
        checkpoint_name = Path(checkpoint_path).name
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"eval_{checkpoint_name}_{timestamp}.json")

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, cls=CustomJSONEncoder, indent=4)

        logger.info("Results saved to: %s", results_file)

        # Also save as CSV for easier analysis
        csv_file = results_file.replace(".json", ".csv")
        df = pd.DataFrame(episode_stats)
        df.to_csv(csv_file, index=False)
        logger.info("Episode data saved to: %s", csv_file)

    # Cleanup
    env.close()
    ray.shutdown()

    return summary_stats


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Ray/RLlib trained models on AdvBuildingGym"
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "ddpg", "td3", "a2c"],
        help="RL algorithm to evaluate"
    )
    parser.add_argument(
        "-cn", "--config-name",
        type=str,
        help="Name of the configuration (used for checkpoint search path)"
    )
    parser.add_argument(
        "--load-config",
        type=str,
        help="Path to JSON config file to load (e.g., 'configs/my_config.json')"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Ray checkpoint directory. If not provided, searches for best checkpoint."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--log-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-step trajectory JSON per episode (default: True)"
    )

    args = parser.parse_args()

    # Load config from file if specified, otherwise use default
    if args.load_config:
        logger.info("Loading config from: %s", args.load_config)
        active_config = ConfigManager.load(args.load_config)
        logger.info("Config loaded successfully: %s", active_config.config_name)
    else:
        active_config = default_config

    # Initialise singleton component instances in the main process before use.
    active_config.init_singletons()

    args.config_name = active_config.config_name if args.config_name is None else args.config_name

    logger.info("Parsed arguments: %s", vars(args))

    # Determine checkpoint path
    if args.checkpoint is None:
        search_base = f"models/{args.config_name}/ray/{args.algorithm}"

        if os.path.exists(search_base):
            # First try to find the best-performing checkpoint via metadata
            try:
                logger.info("Searching for best checkpoint in: %s", search_base)
                checkpoint_path = find_best_checkpoint(search_base)
            except FileNotFoundError:
                # Fallback to latest checkpoint by modification time
                logger.info("No best checkpoint metadata found, falling back to latest checkpoint")
                checkpoint_path = find_latest_checkpoint(search_base)
        else:
            logger.warning("Algorithm directory not found: %s", search_base)
            logger.info("Searching in all models...")
            try:
                checkpoint_path = find_best_checkpoint("models")
            except FileNotFoundError:
                checkpoint_path = find_latest_checkpoint()
    else:
        checkpoint_path = args.checkpoint

    # Convert to absolute path (required by Ray)
    checkpoint_path = os.path.abspath(checkpoint_path)

    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    # Run evaluation
    try:
        results = evaluate_ray_model(
            checkpoint_path=checkpoint_path,
            active_config=active_config,
            num_episodes=args.episodes,
            seed=args.seed,
            save_results=not args.no_save,
            output_dir=args.output_dir,
            log_trajectories=args.log_trajectories,
        )

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error("Evaluation failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Usage examples:
# Evaluate best PPO model with default config
# python run_eval_ray.py --algorithm ppo --episodes 10 --seed 42

# Evaluate with a custom config file (matching training config)
# python run_eval_ray.py --algorithm ppo --load-config configs/my_config.json --episodes 10

# Evaluate latest SAC model
# python run_eval_ray.py --algorithm sac --config-name test1 --episodes 10

# Evaluate specific checkpoint
# python run_eval_ray.py --checkpoint models/test1/ray/ppo/checkpoints_ppo_seed42_20260106/best_model_ep100_...

# Evaluate without saving results
# python run_eval_ray.py --algorithm ppo --episodes 50 --no-save
