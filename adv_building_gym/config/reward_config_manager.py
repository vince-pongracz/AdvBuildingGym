"""Reward config serialization and management utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

import yaml

if TYPE_CHECKING:
    from adv_building_gym.config.reward_config import RewardConfig


class RewardConfigManager:
    """Handles serialization and deserialization of RewardConfig objects.

    Uses the flexible serialization system where each RewardFunction
    knows how to serialize itself via the Serializable mixin.
    """

    @staticmethod
    def to_dict(reward_config: RewardConfig) -> Dict[str, Any]:
        """Serialize RewardConfig to dictionary.

        Args:
            reward_config: RewardConfig object to serialize

        Returns:
            Dictionary with a ``rewards`` key containing serialized reward list.
        """
        config_dict: Dict[str, Any] = {}

        if reward_config.rewards is not None:
            config_dict["rewards"] = [
                reward.to_dict() for reward in reward_config.rewards
            ]

        return config_dict

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> RewardConfig:
        """Deserialize RewardConfig from dictionary.

        Uses the ComponentRegistry to find the correct class for each
        reward and reconstructs it from serialized parameters.

        Args:
            config_dict: Dictionary representation (must contain ``rewards`` key).

        Returns:
            RewardConfig with deserialized reward instances.
        """
        from adv_building_gym.config.reward_config import RewardConfig
        from adv_building_gym.rewards import RewardFunction

        reward_config = RewardConfig()

        if "rewards" in config_dict:
            rewards = []
            for reward_dict in config_dict["rewards"]:
                reward = RewardFunction.from_dict(reward_dict)
                rewards.append(reward)
            reward_config.rewards = rewards

        return reward_config

    @staticmethod
    def save(reward_config: RewardConfig, path: str | Path) -> None:
        """Save RewardConfig to YAML file.

        Args:
            reward_config: RewardConfig object to save
            path: File path (e.g., 'configs/my_rewards.yaml')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = RewardConfigManager.to_dict(reward_config)

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def load(path: str | Path) -> RewardConfig:
        """Load RewardConfig from YAML file.

        Args:
            path: File path to load from (e.g., 'configs/my_rewards.yaml')

        Returns:
            RewardConfig reconstructed from file.
        """
        path = Path(path)

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return RewardConfigManager.from_dict(config_dict)
