"""Serialization for RewardConfig.

Handles conversion to/from dictionaries and YAML files, keeping
RewardConfig itself as a pure data + factory class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import yaml

if TYPE_CHECKING:
    from adv_building_gym.config.rewards.reward_config import RewardConfig

logger = logging.getLogger(__name__)

class RewardConfigSerializer:
    """Handles serialization and deserialization of RewardConfig objects.

    Mirrors the pattern used by EnvConfigManager for EnvConfig.
    """

    @staticmethod
    def to_dict(reward_config: RewardConfig) -> Dict[str, Any]:
        """Serialize a RewardConfig to a dictionary.

        Each RewardFunction serializes itself via the Serializable mixin.

        Args:
            reward_config: RewardConfig object to serialize.

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
        """Deserialize a RewardConfig from a dictionary.

        Uses the ComponentRegistry to find the correct class for each
        reward and reconstructs it from serialized parameters.

        Args:
            config_dict: Dictionary that may contain a ``rewards`` key.

        Returns:
            RewardConfig with deserialized reward instances.
        """
        from adv_building_gym.config.rewards.reward_config import RewardConfig
        from adv_building_gym.components.registry import from_dict as component_from_dict

        reward_config = RewardConfig()
        if "rewards" in config_dict:
            rewards = []
            for reward_dict in config_dict["rewards"]:
                reward = component_from_dict(reward_dict, "reward")
                rewards.append(reward)
            reward_config.rewards = rewards
            reward_config.log_values()
        return reward_config

    @staticmethod
    def save(reward_config: RewardConfig, path: str | Path) -> None:
        """Save a RewardConfig to a YAML file.

        Args:
            reward_config: RewardConfig object to save.
            path: File path (e.g., 'configs/my_rewards.yaml').
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = RewardConfigSerializer.to_dict(reward_config)
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def load(path: str | Path) -> RewardConfig:
        """Load a RewardConfig from a YAML file.

        Args:
            path: File path to load from (e.g., 'configs/my_rewards.yaml').

        Returns:
            RewardConfig reconstructed from file.
        """
        path = Path(path)
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return RewardConfigSerializer.from_dict(config_dict)
