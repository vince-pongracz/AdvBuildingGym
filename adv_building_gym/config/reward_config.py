"""Reward configuration for AdvBuildingGym.

Defines which reward functions participate in the environment and their weights.
Separated from EnvConfig to allow independent reward composition.

Reward functions are fully decoupled from infrastructure — they receive
runtime data (power flows, EV charger params) via the ``info`` dict
passed to ``get_reward()`` by the environment.

Serialization is handled by ``RewardConfigSerializer`` (separation of concerns).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

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

    Serialization: use ``RewardConfigSerializer.save()`` / ``.load()``.
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
            EconomicReward(weight=1),
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
