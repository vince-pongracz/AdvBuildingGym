"""
run_train_sb.py – Train PPO, SAC, DDPG, TD3, or A2C using Stable Baselines3.
This script sets up parallel training and evaluation environments with configurable
settings. It trains an RL agent and saves the model and logs.
"""

import os
import time
import datetime
import argparse
import logging
import json

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from adv_building_gym import AdvBuildingGym
from adv_building_gym.env_config import config as env_config
from adv_building_gym.utils import CustomJSONEncoder

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True
)
logger = logging.getLogger("main")

# TODO VP 2026.01.07. : How does the _model_config look like here with the SB setup? -- compare it with the ray setup

# TODO VP 2026.01.08. : Extract these callbacks into a module
class EpisodeLoggerCallback(BaseCallback):
    """Callback to log episode metrics during training."""

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if any episode finished
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                infos = self.locals.get("infos", [])
                if idx < len(infos):
                    info = infos[idx]
                    episode_info = info.get("episode", {})
                    if episode_info:
                        # SB3's VecMonitor uses short keys: 'r' (reward), 'l' (length), 't' (time)
                        # We normalize these to descriptive names for clarity and consistency.
                        ep_reward = episode_info.get("reward", episode_info.get("r", 0))
                        ep_length = episode_info.get("length", episode_info.get("l", 0))

                        self.episode_count += 1

                        logger.info(
                            "Episode %d ended. Length: %d, Reward: %.2f",
                            self.episode_count, ep_length, ep_reward
                        )
        return True


class CustomEvalCallback(EvalCallback):
    """
    Extended EvalCallback that tracks additional metrics for compatibility with Ray.

    Metrics tracked:
    - episode_return_mean: Mean episode reward (same as best_mean_reward)
    - achieved_reward: Mean cumulative reward per episode
    - reward_rate: Ratio of achieved reward to theoretical maximum
        * Calculated as: achieved_reward / (episode_length × num_reward_functions)
        * Assumes each reward function can contribute 1.0 per timestep
        * Example: 289 timesteps × 3 rewards = 867 max possible reward
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_episode_return_mean = -np.inf
        self.best_achieved_reward = -np.inf
        self.best_reward_rate = -np.inf

        # Calculate max possible reward per episode
        # Assumes each reward function can contribute up to 1.0 per timestep
        # max_achievable_reward = episode_length * num_reward_functions
        self.episode_length = env_config.EPISODE_LENGTH + 1  # ~289 timesteps
        self.num_rewards = len(env_config.rewards)
        self.max_achievable_reward = self.episode_length * self.num_rewards

        logger.info(
            "CustomEvalCallback initialized: episode_length=%d, num_rewards=%d, max_achievable_reward=%.1f",
            self.episode_length, self.num_rewards, self.max_achievable_reward
        )

    def _on_step(self) -> bool:
        """Called after each evaluation."""
        result = super()._on_step()

        # After evaluation completes, calculate additional metrics
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Get episode rewards from the last evaluation
            episode_rewards = self.evaluations_results[-1] if self.evaluations_results else []

            if len(episode_rewards) > 0:
                # episode_return_mean: same as mean_reward
                episode_return_mean = np.mean(episode_rewards)

                # achieved_reward: mean of episode totals (same as episode_return_mean for now)
                # In SB3, episode_rewards are already the cumulative rewards per episode
                achieved_reward = np.mean(episode_rewards)

                # reward_rate: achieved reward / max possible reward
                reward_rate = achieved_reward / self.max_achievable_reward if self.max_achievable_reward > 0 else 0.0

                # Update best values
                if episode_return_mean > self.best_episode_return_mean:
                    self.best_episode_return_mean = episode_return_mean

                if achieved_reward > self.best_achieved_reward:
                    self.best_achieved_reward = achieved_reward

                if reward_rate > self.best_reward_rate:
                    self.best_reward_rate = reward_rate

                if self.verbose > 0:
                    logger.info(
                        "Eval metrics - episode_return_mean: %.4f, achieved_reward: %.4f, reward_rate: %.4f (max_possible: %.1f)",
                        episode_return_mean, achieved_reward, reward_rate, self.max_achievable_reward
                    )

        return result


class BestModelCheckpointCallback(BaseCallback):
    """
    Callback to save checkpoints every N episodes and keep only the best model.

    Similar to Ray Tune's CheckpointConfig, this callback:
    - Saves checkpoints every checkpoint_frequency episodes
    - Tracks the best model based on evaluation metric
    - Saves new best model first, then deletes old one
    - Keeps only num_to_keep best checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: str,
        eval_callback: CustomEvalCallback,
        checkpoint_frequency: int = 20,
        num_to_keep: int = 1,
        metric: str = "mean_reward",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.eval_callback = eval_callback
        self.checkpoint_frequency = checkpoint_frequency
        self.num_to_keep = num_to_keep
        self.metric = metric
        self.best_metric_value = -np.inf
        self.current_best_path = None
        self.episode_count = 0
        self.last_checkpoint_episode = 0

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """Check if we should save a checkpoint after episode completion."""
        # Check if any episode finished in this step
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.episode_count += 1

                # Check if we should checkpoint at this episode count
                if self.episode_count % self.checkpoint_frequency == 0:
                    # Avoid multiple checkpoints at the same episode count
                    # (multiple envs can finish simultaneously)
                    if self.episode_count != self.last_checkpoint_episode:
                        self._save_checkpoint()
                        self.last_checkpoint_episode = self.episode_count

        return True

    def _save_checkpoint(self) -> None:
        """Save checkpoint if current model is better than previous best."""
        # Get the latest evaluation score based on selected metric
        # Note: "mean_reward" is normalized to "episode_return_mean" at CLI parsing
        if self.metric == "episode_return_mean":
            # SB3's best_mean_reward is equivalent to Ray's episode_return_mean
            current_metric_value = self.eval_callback.best_mean_reward
        elif self.metric == "achieved_reward":
            current_metric_value = self.eval_callback.best_achieved_reward
        elif self.metric == "reward_rate":
            current_metric_value = self.eval_callback.best_reward_rate
        else:
            # Default to episode_return_mean if unknown metric
            logger.warning(
                "Unknown metric '%s', using 'episode_return_mean' instead",
                self.metric
            )
            current_metric_value = self.eval_callback.best_mean_reward

        logger.info(
            "Episode %d: Checking checkpoint (current_%s=%.4f, best_%s=%.4f)",
            self.episode_count, self.metric, current_metric_value,
            self.metric, self.best_metric_value
        )

        # Check if this is a new best model
        if current_metric_value > self.best_metric_value:
            # Save old checkpoint path for deletion after new one is saved
            old_best_path = self.current_best_path

            # Save new best checkpoint FIRST
            checkpoint_name = f"best_model_ep{self.episode_count}_{self.metric}{current_metric_value:.4f}"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

            self.model.save(checkpoint_path)
            self.best_metric_value = current_metric_value
            self.current_best_path = checkpoint_path

            logger.info(
                "New best model saved at episode %d with %s %.4f: %s",
                self.episode_count, self.metric, current_metric_value, checkpoint_path + ".zip"
            )

            # Save metadata
            metadata = {
                "episode": self.episode_count,
                "metric": self.metric,
                "best_metric_value": float(current_metric_value),
                "num_timesteps": self.model.num_timesteps,
                "checkpoint_path": checkpoint_path + ".zip",
            }
            metadata_path = os.path.join(self.checkpoint_dir, "best_checkpoint_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

            # Now delete old best checkpoint AFTER new one is safely saved
            if old_best_path is not None and os.path.exists(old_best_path + ".zip"):
                logger.info("Deleting old best checkpoint: %s", old_best_path + ".zip")
                try:
                    os.remove(old_best_path + ".zip")
                except OSError as e:
                    logger.warning("Failed to delete old checkpoint: %s", e)


def make_env(rank: int, seed: int):
    """Factory function for creating environment instances."""
    def _init() -> gym.Env:
        env = AdvBuildingGym(
            infras=env_config.infras,
            datasources=env_config.datasources,
            rewards=env_config.rewards,
            building_props=env_config.building_props,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def select_model(algorithm: str, env, seed: int, device: torch.device):
    """
    Selects and configures a Stable Baselines3 model based on the algorithm name.
    """
    # Hyperparameters for consistent evaluation across algorithms
    learning_rate = 3e-4
    batch_size = 64
    buffer_size = 100_000
    learning_starts = 10 * env_config.EPISODE_LENGTH
    n_steps = env_config.EPISODE_LENGTH

    common_kwargs = {
        "policy": "MultiInputPolicy",
        "env": env,
        "verbose": 1,
        "seed": seed,
        "device": device,
        "learning_rate": learning_rate,
    }

    if algorithm == "ppo":
        model = PPO(
            **common_kwargs,
            n_steps=n_steps,
            batch_size=batch_size,
        )
    elif algorithm == "sac":
        model = SAC(
            **common_kwargs,
            learning_starts=learning_starts,
            batch_size=batch_size,
            buffer_size=buffer_size,
            use_sde=True,
        )
    elif algorithm == "ddpg":
        model = DDPG(
            **common_kwargs,
            learning_starts=learning_starts,
            batch_size=batch_size,
            buffer_size=buffer_size,
        )
    elif algorithm == "td3":
        model = TD3(
            **common_kwargs,
            learning_starts=learning_starts,
            batch_size=batch_size,
            buffer_size=buffer_size,
        )
    elif algorithm == "a2c":
        model = A2C(
            **common_kwargs,
            n_steps=n_steps,
            use_sde=True,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model


def main():
    """Parse CLI arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train RL agents using Stable Baselines3"
    )
    parser.add_argument(
        "--algorithm", "-a",
        default="ppo",
        choices=["ppo", "sac", "ddpg", "td3", "a2c"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--config_name", "-cn",
        type=str,
        default=None,
        help="Name of the configuration/experiment"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=float,
        default=1e6,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--num-envs", "-n",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=20_000,
        help="Evaluation frequency in timesteps"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="reward_rate",
        choices=[
            # Both "mean_reward" and "episode_return_mean" are accepted for compatibility
            # (SB3 uses "mean_reward", Ray uses "episode_return_mean" - they are equivalent)
            "mean_reward",
            "episode_return_mean",
            "achieved_reward",
            "reward_rate",
        ],
        help="Metric to optimize during training for checkpoint selection",
    )
    parser.add_argument(
        "--checkpoint-frequency-episodes",
        type=int,
        default=20,
        help="Checkpoint frequency in number of episodes",
    )
    args = parser.parse_args()

    args.timesteps = int(args.timesteps)
    args.config_name = env_config.config_name if args.config_name is None else args.config_name

    # Normalize metric name: SB3's "mean_reward" is equivalent to Ray's "episode_return_mean"
    # Use "episode_return_mean" internally for consistency across frameworks
    if args.metric == "mean_reward":
        args.metric = "episode_return_mean"

    logger.info("Parsed arguments: %s", vars(args))

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Create directories
    exec_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algorithm}_seed{args.seed}_{exec_date}"
    base_path = f"models/{args.config_name}/sb3/{args.algorithm}"
    model_save_path = f"{base_path}/{run_name}"
    best_model_path = f"{base_path}/best_{run_name}"
    checkpoint_dir = f"{base_path}/checkpoints_{run_name}"
    log_path = f"logs/{args.config_name}/sb3/{run_name}"

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save run configuration
    config_dump = {
        "algorithm": args.algorithm,
        "config_name": args.config_name,
        "timesteps": args.timesteps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "eval_freq": args.eval_freq,
        "device": str(device),
        "run_name": run_name,
        "infras": [str(i) for i in env_config.infras],
        "rewards": [str(r) for r in env_config.rewards],
    }
    with open(f"{model_save_path}/config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, cls=CustomJSONEncoder, indent=4)

    # Training environment setup
    logger.info("Creating %d training environments...", args.num_envs)
    if args.num_envs > 1:
        train_env = SubprocVecEnv(
            [make_env(rank=i, seed=args.seed) for i in range(args.num_envs)]
        )
    else:
        train_env = DummyVecEnv([make_env(rank=0, seed=args.seed)])
    train_env = VecMonitor(train_env)

    logger.info("Observation space: %s", train_env.observation_space)
    logger.info("Action space: %s", train_env.action_space)

    # Callbacks
    callbacks = [EpisodeLoggerCallback(log_dir=log_path)]

    # Evaluation environment setup (if enabled)
    eval_env = None
    eval_callback = None
    if not args.no_eval:
        logger.info("Creating evaluation environment...")
        eval_env = DummyVecEnv([make_env(rank=0, seed=args.seed + 10_000)])
        eval_env = VecMonitor(eval_env)

        eval_callback = CustomEvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=log_path,
            eval_freq=max(args.eval_freq // args.num_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callbacks.append(eval_callback)

        # Add checkpoint callback that saves best model based on configured frequency
        checkpoint_callback = BestModelCheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            eval_callback=eval_callback,
            checkpoint_frequency=args.checkpoint_frequency_episodes,
            num_to_keep=1,
            metric=args.metric,
            verbose=1,
        )
        callbacks.append(checkpoint_callback)
        logger.info(
            "Checkpoint callback enabled: saving best model every %d episodes based on '%s' metric to %s",
            args.checkpoint_frequency_episodes, args.metric, checkpoint_dir
        )

    # Create and train model
    logger.info("Creating %s model...", args.algorithm.upper())
    model = select_model(args.algorithm, train_env, args.seed, device)

    logger.info("Starting training for %d timesteps...", args.timesteps)
    t0 = time.time()

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    training_time = (time.time() - t0) / 60
    logger.info("Training completed in %.2f min", training_time)

    # Save final model
    final_model_path = f"{model_save_path}/final_model"
    model.save(final_model_path)
    logger.info("Final model saved to: %s", final_model_path)

    if not args.no_eval:
        logger.info("Best model saved to: %s", best_model_path)
        logger.info("Best checkpoint saved to: %s", checkpoint_dir)

    # Save training summary
    summary = {
        "training_time_min": training_time,
        "final_model_path": final_model_path,
        "best_model_path": best_model_path if not args.no_eval else None,
        "checkpoint_dir": checkpoint_dir if not args.no_eval else None,
        "total_timesteps": args.timesteps,
    }
    with open(f"{model_save_path}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Cleanup
    train_env.close()
    if eval_env is not None:
        eval_env.close()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

# Usage examples:
# Default settings (checkpoint every 20 episodes, optimize mean_reward)
# python run_train_sb.py --algorithm ppo --seed 42 --timesteps 1e6

# Custom checkpoint frequency and metric
# python run_train_sb.py -a ppo -s 42 -t 1e6 --checkpoint-frequency-episodes 50 --metric reward_rate

# Multiple environments with custom settings
# python run_train_sb.py -a sac -s 18 -t 500000 -n 8 --checkpoint-frequency-episodes 30

# Training without evaluation/checkpoints
# python run_train_sb.py --algorithm td3 --no-eval --timesteps 100000
