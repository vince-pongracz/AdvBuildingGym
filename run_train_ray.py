

"""
run_train_rl.py – Train PPO, SAC, DDPG, TD3, or A2C on LLEC-HeatPumpHouse environments.
This script sets up parallel training and evaluation environments with configurable observation
and reward settings. It trains an RL agent using Stable Baselines3 and saves the model and logs.
"""

import os
import sys
import time
import logging

from pprint import pprint

import json
import argparse
import torch
import gymnasium as gym

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

os.environ["PYTHONWARNINGS"] ="ignore::DeprecationWarning"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

# Trigger registration of the custom Gym IDs
# TODO VP 2025.12.16. : rename building_gym things...
from llec_building_gym import AdvBuildingGym
from llec_building_gym.env_config import config as env_config

from llec_building_gym.utils import CustomJSONEncoder

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(logger) - %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug("sys.argv: %s", sys.argv)

episode_length:int = 288

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.policy.policy import Policy
from typing import Dict, List
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

class Callbacks(RLlibCallback):
    def on_episode_end(self,
                       *,
                       episode: SingleAgentEpisode | MultiAgentEpisode,
                       prev_episode_chunks: List[SingleAgentEpisode | MultiAgentEpisode] | None = None,
                       env_runner: EnvRunner | None = None,
                       metrics_logger: MetricsLogger | None = None,
                       env: gym.Env | None = None,
                       env_index: int,
                       rl_module: RLModule | None = None,
                       worker: EnvRunner | None = None,
                       base_env: BaseEnv | None = None,
                       policies: Dict[str, Policy] | None = None,
                       **kwargs) -> None:
        episode.custom_metrics["max_possible_reward_per_episode"] = len(env_config.rewards) * episode.length
        episode.custom_metrics["episode_length"] = episode.length
        return super().on_episode_end(episode=episode, prev_episode_chunks=prev_episode_chunks, env_runner=env_runner, metrics_logger=metrics_logger, env=env, env_index=env_index, rl_module=rl_module, worker=worker, base_env=base_env, policies=policies, **kwargs)



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
    learning_starts = 10 * episode_length
    n_steps = episode_length
    
    if algorithm == "ppo":
        algo_config = (PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .learners(
                num_learners=1,
                num_gpus_per_learner=1,
                num_cpus_per_learner=1,
            )
            .environment(
                env="AdvBuilding",
            )
            .training(
                lr=learning_rate,
                train_batch_size_per_learner=2000,
                num_epochs=n_steps,
                model={
                    "fcnet_hiddens": [256, 256],
                }
            )
            .debugging(log_level="ERROR")
            .framework(
                framework="torch",
                eager_tracing=True,
                eager_max_retraces=20,
                tf_session_args={},
                local_tf_session_args={},
            )
            .callbacks(Callbacks)
        )
        algo_config.logger_config = {
            "type": "ray.tune.logger.UnifiedLogger",
            "loggers": [
                "ray.tune.json.JsonLoggerCallback",
                "ray.tune.csv.CSVLoggerCallback",
                "ray.tune.tensorboardx.TBXLoggerCallback",
            ],
        }
        algo = algo_config.build_algo()
    # TODO VP 2025.12.11. : add more RL algos
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return algo


def env_creator(config):
    return AdvBuildingGym(
        infras=env_config.infras,
        datasources=env_config.datasources,
        rewards=env_config.rewards,
        building_props=env_config.building_props,
    )  #Return a gymnasium.Env instance.


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
    parser.add_argument("--timesteps", type=float, default=1e6)
    # TODO VP 2025.12.09. : change env number
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument(
        "--training",
        action="store_true",
        help="Use training split of price data (else test split)",
    )
    args = parser.parse_args()

    args.timesteps = int(args.timesteps)
    args.config_name = env_config.config_name if args.config_name is None else args.config_name
    logger.info("Parsed arguments: %s", vars(args))

    # TODO VP 2025.12.17. : these settings should come from the slurm config / as args.
    ray.init(num_cpus=2, ignore_reinit_error=True)
    
    logger.info("Ray initialized.")
    # Map usecase to environment Gym Environment ID alias per reward mode (registered by llec_building_gym)
    env_id:str = "AdvBuilding"
    tune.register_env(env_id, env_creator)
    # Training Model
    # TODO VP 2025.12.17. : fix this None thing here
    algo = select_model(args.algorithm, None, args.seed)

    t0 = time.time()
    for i in range(1):
        result = algo.train()

        result['max_possible_reward'] = episode_length * len(env_config.rewards)
        result["reward_rate"] = result['env_runners']['episode_reward_mean'] / result['max_possible_reward']
        
        logger.info(
            f"""
            iter={i+1}
            reward_mean={result['env_runners']['episode_reward_mean']:.2f}
            reward_rate={result['reward_rate']:.4f}
            timesteps_total={result['timesteps_total']}
            """)
        with open(f"result_{i}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, cls=CustomJSONEncoder, indent=4)

    # Log training end time, duration, and save location
    logger.info("Training completed in %.2f min", (time.time() - t0) / 60)
    
    # Saving Model
    save_dir = f"models/{args.config_name}/best_{args.algorithm}/ray"
    os.makedirs(save_dir, exist_ok=True)
    save_path: str = f"{save_dir}/{args.algorithm}_model_seed{args.seed}"
    algo.save(save_path)
    
    logger.info("Model saved to: %s", save_path)
    logger.info("Best model path: models/%s/best_%s/", args.config_name, args.algorithm)

    # Release the algo's resources (remote actors, like EnvRunners and Learners).
    algo.stop()
    ray.shutdown()
    

if __name__ == "__main__":
    main()

# On slurm: sbatch slurm_script/slurm_train_ray.sh
# On desktop: python run_train_v2.py --algorithm ppo --training --seed 18 --timesteps 1e6
