"""
run_train_rl.py – Train PPO, SAC, DDPG, TD3, or A2C on LLEC-HeatPumpHouse environments.
This script sets up parallel training and evaluation environments with configurable observation
and reward settings. It trains an RL agent using Stable Baselines3 and saves the model and logs.
"""

import os
import sys
import time
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Trigger registration of the custom Gym IDs
from adv_building_gym import BaseBuildingGym
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True
)
logger = logging.getLogger("main")


# Helper classes & functions
class SeedWrapper(gym.Wrapper):
    """Ensures that env.reset() is called with a fixed seed unless
    the user explicitly provides one (Gymnasium > 0.29 signature)."""

    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=self._seed if seed is None else seed, options=options)


def make_env(env_id: str, rank: int, base_seed: int, **env_kwargs):
    """Factory function for SubprocVecEnv."""

    def _init() -> gym.Env:
        env = gym.make(env_id, **env_kwargs)
        env = SeedWrapper(env, seed=base_seed + rank)
        return env

    return _init


def select_model(algorithm: str, env: gym.Env, seed: int):
    """
    Selects and configures a model for training based on the algorithm name.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Hyperparameters for consistent evaluation across algorithms
    learning_rate = 3e-4
    batch_size = 64
    buffer_size = 100_000
    episode_length = 288
    learning_starts = 10 * episode_length
    n_steps = episode_length
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

    elif algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_starts=learning_starts,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            device=device,
            use_sde=True,
        )

    elif algorithm == "ddpg":
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_starts=learning_starts,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            device=device,
        )

    elif algorithm == "td3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_starts=learning_starts,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            device=device,
        )

    elif algorithm == "a2c":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            n_steps=n_steps,
            learning_rate=learning_rate,
            device=device,
            use_sde=True,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return model


# Main
def main():
    """Parse CLI arguments and run training for selected algorithms."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", default="ppo", choices=["ppo", "sac", "ddpg", "a2c"]
    )
    parser.add_argument("--timesteps", type=float, default=1e6)
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument(
        "--reward_mode", default="temperature", choices=["temperature", "combined"]
    )
    parser.add_argument("--energy-price-path", default="data/price_data_2025.csv")
    parser.add_argument(
        "--training",
        action="store_true",
        help="Use training split of price data (else test split)",
    )
    parser.add_argument(
        "--obs_variant",
        default="T01",
        choices=["T01", "T02", "T03", "T04", "C01", "C02", "C03", "C04"],
    )
    args = parser.parse_args()

    args.timesteps = int(args.timesteps)
    logger.info("Parsed arguments: %s", vars(args))

    # Map usecase to environment Gym Environment ID alias per reward mode (registered by llec_building_gym)
    env_id = {
        "temperature": "LLEC-HeatPumpHouse-1R1C-Temperature-v0",
        "combined": "LLEC-HeatPumpHouse-1R1C-Combined-v0",
    }[args.reward_mode]

    # Training environments setup
    train_env = SubprocVecEnv(
        [
            make_env(
                env_id,
                rank=i,
                base_seed=args.seed,
                energy_price_path=args.energy_price_path,
                training=args.training,
                schedule_type=None,
                obs_variant=args.obs_variant,
            )
            for i in range(args.num_envs)
        ]
    )
    train_env = VecMonitor(train_env)

    # Evaluation environments setup
    eval_env = SubprocVecEnv(
        [
            make_env(
                env_id,
                rank=i,
                # ensure no overlap with training seeds
                base_seed=args.seed + 10_000,
                energy_price_path=args.energy_price_path,
                training=False,
                schedule_type=None,
                obs_variant=args.obs_variant,
            )
            for i in range(args.num_envs)
        ]
    )
    eval_env = VecMonitor(eval_env)

    # Evaluation callback setup
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{args.reward_mode}/{args.obs_variant}/best_{args.algorithm}/",
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        deterministic=True,
        render=False,
    )
    # Training Model
    model = select_model(args.algorithm, train_env, args.seed)
    logger.info("Observation space: %s", train_env.observation_space)
    logger.info("Action space:      %s", train_env.action_space)
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True
    )
    # Log training end time, duration, and save location
    logger.info("Training completed in %.2f min", (time.time() - t0) / 60)

    # Saving Model
    save_dir = f"models/{args.reward_mode}/{args.obs_variant}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{args.algorithm}_model_seed{args.seed}"
    model.save(save_path)
    logger.info("Model saved to: %s", save_path)
    logger.info(
        f"Best model path: models/{args.reward_mode}/{args.obs_variant}/best_{args.algorithm}/"
    )

    # Close environments
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

# python run_train_rl.py --algorithm ppo --reward_mode temperature --obs_variant T01 --training --seed 18 --timesteps 1e6
