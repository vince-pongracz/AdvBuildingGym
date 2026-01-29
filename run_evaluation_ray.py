"""
run_evaluation_ray.py – Evaluate Ray/RLlib trained models on AdvBuildingGym.

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

from adv_building_gym import AdvBuildingGym
from adv_building_gym.config import config as env_config
from adv_building_gym.utils import CustomJSONEncoder, setup_warning_filters

# Apply warning filters
setup_warning_filters()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True
)
logger = logging.getLogger("main")


def find_latest_checkpoint(base_path: str = "models") -> str:
    """
    Find the most recent Ray checkpoint in the models directory.

    Returns:
        str: Path to the checkpoint directory
    """
    checkpoint_paths = []

    for root, dirs, files in os.walk(base_path):
        if "checkpoint_" in root and any(f.endswith(".pkl") for f in files):
            # Get modification time
            mtime = os.path.getmtime(root)
            checkpoint_paths.append((mtime, root))

    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {base_path}")

    # Sort by modification time and return most recent
    checkpoint_paths.sort(reverse=True)
    latest_checkpoint = checkpoint_paths[0][1]

    logger.info("Found %d checkpoints, using latest: %s", len(checkpoint_paths), latest_checkpoint)
    return latest_checkpoint


def evaluate_ray_model(
    checkpoint_path: str,
    num_episodes: int = 10,
    seed: int = 42,
    save_results: bool = True,
    output_dir: str = "eval_results",
):
    """
    Evaluate a Ray/RLlib trained model on AdvBuildingGym.

    Args:
        checkpoint_path: Path to the Ray checkpoint directory
        num_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        save_results: Whether to save results to file
        output_dir: Directory to save evaluation results

    Returns:
        dict: Evaluation statistics
    """
    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = 1 if torch.cuda.is_available() else 0

    logger.info("=" * 70)
    logger.info("Starting Ray model evaluation")
    logger.info("  Checkpoint: %s", checkpoint_path)
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

    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    env = AdvBuildingGym(
        infras=env_config.infras,
        statesources=env_config.statesources,
        rewards=env_config.rewards,
        building_props=env_config.building_props,
        training=False,  # Evaluation mode
    )

    # Evaluation loop
    episode_stats = []
    all_rewards = []
    start_time = time.time()

    for ep in range(num_episodes):
        logger.info("=" * 50)
        logger.info("Episode %d/%d", ep + 1, num_episodes)

        obs, info = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_rewards = []

        while not done:
            # Compute action using the policy (inference mode)
            # Note: RLlib expects dict observations, which AdvBuildingGym provides
            action = algo.compute_single_action(obs, explore=False)

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            episode_rewards.append(reward)

            obs = next_obs

        # Calculate episode statistics
        achieved_reward = np.sum(episode_rewards)
        max_achievable_reward = episode_length * len(env_config.rewards)
        reward_rate = achieved_reward / max_achievable_reward if episode_length > 0 else 0.0

        ep_stats = {
            "episode": ep + 1,
            "length": episode_length,
            "total_reward": float(episode_reward),
            "achieved_reward": float(achieved_reward),
            "max_achievable_reward": float(max_achievable_reward),
            "reward_rate": float(reward_rate),
            "seed": seed + ep,
        }

        episode_stats.append(ep_stats)
        all_rewards.append(episode_reward)

        logger.info("  Length: %d", episode_length)
        logger.info("  Total Reward: %.2f", episode_reward)
        logger.info("  Achieved Reward: %.2f", achieved_reward)
        logger.info("  Reward Rate: %.4f", reward_rate)

    eval_time = time.time() - start_time

    # Compute summary statistics
    summary_stats = {
        "checkpoint_path": checkpoint_path,
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
    logger.info("  Mean Reward: %.2f ± %.2f", summary_stats["mean_reward"], summary_stats["std_reward"])
    logger.info("  Mean Reward Rate: %.4f ± %.4f", summary_stats["mean_reward_rate"], summary_stats["std_reward_rate"])
    logger.info("  Min/Max Reward: %.2f / %.2f", summary_stats["min_reward"], summary_stats["max_reward"])
    logger.info("  Evaluation time: %.2f seconds", eval_time)
    logger.info("=" * 70)

    # Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename based on checkpoint
        checkpoint_name = Path(checkpoint_path).parent.name
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
        "--config-name", "-cn",
        type=str,
        default="test1",
        help="Configuration name"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Ray checkpoint directory. If not provided, searches for latest checkpoint."
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

    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint is None:
        # Search for latest checkpoint in the algorithm's directory
        search_base = f"models/{args.config_name}/ray/{args.algorithm}"

        if os.path.exists(search_base):
            logger.info("Searching for latest checkpoint in: %s", search_base)
            checkpoint_path = find_latest_checkpoint(search_base)
        else:
            logger.warning("Algorithm directory not found: %s", search_base)
            logger.info("Searching for latest checkpoint in all models...")
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
            num_episodes=args.episodes,
            seed=args.seed,
            save_results=not args.no_save,
            output_dir=args.output_dir,
        )

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error("Evaluation failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Usage examples:
# Evaluate latest PPO model with default config
# python run_evaluation_ray.py --algorithm ppo --episodes 10 --seed 42

# Evaluate latest SAC model
# python run_evaluation_ray.py --algorithm sac --config-name test1 --episodes 10

# Evaluate specific checkpoint (with timestamp and trial_id in path)
# python run_evaluation_ray.py --checkpoint models/test1/ray/ppo/ppo_seed42_20260106_143022/d1e85a/checkpoint_000000

# Evaluate without saving results
# python run_evaluation_ray.py --algorithm ppo --episodes 50 --no-save
