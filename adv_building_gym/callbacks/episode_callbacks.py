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
        ep_length = len(episode)
        ep_achieved_reward = np.sum(episode.get_rewards())
        # Max reward per step = sum of reward weights (assumes base max is 1.0 per reward)
        max_reward_per_step = sum(r.weight for r in rewards)
        max_achievable_reward = ep_length * max_reward_per_step
        reward_rate = ep_achieved_reward / max_achievable_reward if max_achievable_reward > 0 else 0.0

        # Register custom metrics with RLlib's metrics system
        # These will appear in results under "env_runners/achieved_reward_mean" etc.
        metrics_logger.log_value("achieved_reward", ep_achieved_reward, reduce="mean")
        metrics_logger.log_value("reward_rate", reward_rate, reduce="mean")
        # Also log min/max for analysis
        metrics_logger.log_value("achieved_reward_min", ep_achieved_reward, reduce="min")
        metrics_logger.log_value("achieved_reward_max", ep_achieved_reward, reduce="max")
        metrics_logger.log_value("reward_rate_min", reward_rate, reduce="min")
        metrics_logger.log_value("reward_rate_max", reward_rate, reduce="max")

        episode_id: str = episode.id_[:6]
        logger.info(
            "Episode %s ended. Length: %s, Achieved Reward: %.2f, Reward Rate: %.4f",
            episode_id, ep_length, ep_achieved_reward, reward_rate
        )

        # Seconds are not used in the file names
        ep_metrics_dir = f"{metrics_base_dir}/{exec_date.strftime('%Y%m%d_%H%M')}00"
        os.makedirs(ep_metrics_dir, exist_ok=True)
        ep_metrics_file_name = f"{ep_metrics_dir}/episode_{episode_id}_metrics.json"

        # Extract clipped actions from episode infos (actions are clipped before env.step())
        # episode.get_actions() returns raw policy outputs, but we want the clipped ones
        clipped_actions = []
        if hasattr(episode, "get_infos"):
            infos = episode.get_infos()
            if infos and len(infos) > 0:
                # Extract the "clipped_action" field from each info dict (flat clipped array)
                for info in infos:
                    if isinstance(info, dict):
                        if "clipped_action" in info:
                            # Use the flat clipped action stored by the environment
                            clipped_act = info["clipped_action"]
                            if isinstance(clipped_act, np.ndarray):
                                clipped_actions.append(clipped_act.flatten().tolist())
                            else:
                                clipped_actions.append(clipped_act)
                        elif "action" in info:
                            # Fallback: extract from dict action format (for backward compatibility)
                            action_dict = info["action"]
                            if isinstance(action_dict, dict):
                                flat_action = []
                                for key in sorted(action_dict.keys()):
                                    act_val = action_dict[key]
                                    if isinstance(act_val, np.ndarray):
                                        flat_action.extend(act_val.flatten().tolist())
                                    else:
                                        flat_action.append(float(act_val))
                                clipped_actions.append(flat_action)
                            elif isinstance(action_dict, np.ndarray):
                                clipped_actions.append(action_dict.flatten().tolist())

        # If extraction failed, fall back to raw policy actions (unclipped)
        if not clipped_actions:
            logger.warning(
                "Episode %s: Failed to extract clipped actions from infos, using raw policy outputs. "
                "Actions in metrics may exceed [-1, 1] range.",
                episode_id
            )
            clipped_actions = episode.get_actions() if hasattr(episode, "get_actions") else None

        # Build a JSON-serializable dict with episode information.
        dump = {
            "id": episode_id,
            "env_id": env_id,
            "length": ep_length,
            # NOTE VP 2026.01.12. : episode_return_mean is not available here, only in result dict.
            "achieved_reward": float(ep_achieved_reward),
            "total_reward": float(max_achievable_reward),
            "reward_rate": float(reward_rate),
            "rewards": episode.get_rewards() if hasattr(episode, "get_rewards") else None,
            "observations": episode.get_observations() if hasattr(episode, "get_observations") else None,
            "actions": clipped_actions,
            "raw_policy_actions": episode.get_actions() if hasattr(episode, "get_actions") else None,  # For debugging
        }

        with open(ep_metrics_file_name, "w", encoding="utf-8") as f:
            json.dump(dump, f, cls=CustomJSONEncoder, indent=4)

    return on_episode_end
