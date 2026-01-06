"""
Episode-end callbacks for Ray RLlib training.
"""

import os
import json
import logging
import datetime
from typing import Optional, List

import numpy as np

from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from ..utils import CustomJSONEncoder

logger = logging.getLogger(__name__)


def create_on_episode_end_callback(
    env_id: str,
    rewards: List,
    metrics_base_dir: str = "ep_metrics",
    exec_date: Optional[datetime.datetime] = None,
):
    """
    Factory function to create an on_episode_end callback for Ray RLlib.

    Args:
        env_id: Environment identifier for logging.
        rewards: List of reward objects (used to calculate max reward).
        metrics_base_dir: Base directory for saving episode metrics.
        exec_date: Execution datetime for directory naming. Defaults to now.

    Returns:
        A callback function compatible with RLlib's callbacks_on_episode_end.
    """
    if exec_date is None:
        exec_date = datetime.datetime.now()

    def on_episode_end(
        *,
        episode: SingleAgentEpisode,
        env_runner,
        metrics_logger,
        env,
        **kwargs,
    ):
        """
        Custom callback to log additional metrics at the end of each episode.

        Uses metrics_logger to register custom metrics that will be aggregated
        and available in Tune results for optimization (e.g., tune.TuneConfig.metric).
        """
        # Calculate episode metrics
        ep_len = len(episode)
        achieved_reward = np.sum(episode.get_rewards())
        max_achievable_reward = ep_len * len(rewards)
        reward_rate = achieved_reward / max_achievable_reward if ep_len > 0 else 0.0

        # Register custom metrics with RLlib's metrics system
        # These will appear in results under "env_runners/achieved_reward_mean" etc.
        metrics_logger.log_value("achieved_reward", achieved_reward, reduce="mean")
        metrics_logger.log_value("reward_rate", reward_rate, reduce="mean")
        # Also log min/max for analysis
        metrics_logger.log_value("achieved_reward_min", achieved_reward, reduce="min")
        metrics_logger.log_value("achieved_reward_max", achieved_reward, reduce="max")
        metrics_logger.log_value("reward_rate_min", reward_rate, reduce="min")
        metrics_logger.log_value("reward_rate_max", reward_rate, reduce="max")

        episode_id: str = episode.id_[:6]
        logger.info(
            "Episode %s ended. Length: %s, Achieved Reward: %.2f, Reward Rate: %.4f",
            episode_id, ep_len, achieved_reward, reward_rate
        )

        ep_metrics_dir = f"{metrics_base_dir}/{exec_date.strftime('%Y%m%d_%H%M')}"
        os.makedirs(ep_metrics_dir, exist_ok=True)
        ep_metrics_file_name = f"{ep_metrics_dir}/episode_{episode_id}_metrics.json"

        # Build a JSON-serializable dict with meaningful episode information.
        dump = {
            "id": episode_id,
            "env_id": env_id,
            "length": ep_len,
            "achieved_reward": float(achieved_reward),
            "total_reward": float(max_achievable_reward),
            "reward_rate": float(reward_rate),
            "rewards": episode.get_rewards() if hasattr(episode, "get_rewards") else None,
            "observations": episode.get_observations() if hasattr(episode, "get_observations") else None,
            "actions": episode.get_actions() if hasattr(episode, "get_actions") else None,
        }

        with open(ep_metrics_file_name, "w", encoding="utf-8") as f:
            json.dump(dump, f, cls=CustomJSONEncoder, indent=4)

    return on_episode_end
