"""Reward configuration for AdvBuildingGym.

Defines which reward functions participate in the environment and their weights.
Separated from EnvConfig to allow independent reward composition.

Reward functions are fully decoupled from infrastructure — they receive
runtime data (power flows, EV charger params) via the ``info`` dict
passed to ``get_reward()`` by the environment.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from adv_building_gym.rewards import (
    RewardFunction, ActionSmoothnessReward, BatteryTargetReward, TempReward,
    EconomicReward, EVChargingOnTimeReward, EVChargingReward,
    MinimiseEnergyConsumptionReward, UserEnergyNeedReward,
    OperatorEnergyControlReward
)

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward function composition.

    Defines which reward functions are active and their weights.
    Use create_rewards() factory method for parallel environments.
    Direct access to self.rewards returns the cached singleton (for inspection only).
    """

    # Cached singleton instances (for backward compatibility and inspection)
    # WARNING: Do not pass these to parallel environments - use factory method instead
    rewards: Optional[List[RewardFunction]] = None

    def create_rewards(self) -> List[RewardFunction]:
        """Factory method to create fresh RewardFunction instances.

        Each call returns NEW independent instances, safe for parallel environments.

        Returns:
            List of newly created RewardFunction instances.
        """
        return [
            TempReward(weight=1, diff_threshold=0.0001),
            EconomicReward(weight=1, max_power_kW=25.0),
            MinimiseEnergyConsumptionReward(weight=0.2),
            OperatorEnergyControlReward(weight=1),
            BatteryTargetReward(weight=1),
            EVChargingReward(weight=1),
            EVChargingOnTimeReward(weight=1),
            # ActionSmoothnessReward(weight=0.5),
        ]

    def init_singletons(self) -> None:
        """Initialise the cached singleton reward instances.

        Call this once in the main process after creating / loading a config.
        Not needed in Ray worker subprocesses — they call create_rewards() directly.
        """
        if self.rewards is None:
            self.rewards = self.create_rewards()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this RewardConfig to a dictionary.

        Each RewardFunction serializes itself via the Serializable mixin.

        Returns:
            Dictionary with a ``rewards`` key containing serialized reward list.
        """
        config_dict: Dict[str, Any] = {}
        if self.rewards is not None:
            config_dict["rewards"] = [
                reward.to_dict() for reward in self.rewards
            ]
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RewardConfig":
        """Deserialize a RewardConfig from a dictionary.

        Uses the ComponentRegistry to find the correct class for each
        reward and reconstructs it from serialized parameters.

        Args:
            config_dict: Dictionary that may contain a ``rewards`` key.

        Returns:
            RewardConfig with deserialized reward instances.
        """
        reward_config = cls()
        if "rewards" in config_dict:
            rewards = []
            for reward_dict in config_dict["rewards"]:
                reward = RewardFunction.from_dict(reward_dict)
                rewards.append(reward)
            reward_config.rewards = rewards
        return reward_config

    def save(self, path: str | Path) -> None:
        """Save this RewardConfig to a YAML file.

        Args:
            path: File path (e.g., 'configs/my_rewards.yaml').
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "RewardConfig":
        """Load a RewardConfig from a YAML file.

        Args:
            path: File path to load from (e.g., 'configs/my_rewards.yaml').

        Returns:
            RewardConfig reconstructed from file.
        """
        path = Path(path)
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
