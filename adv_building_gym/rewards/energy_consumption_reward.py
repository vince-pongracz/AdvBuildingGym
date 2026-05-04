import logging

import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# NOTE VP 2026.01.14. : Add battery life saving reward

class MinimiseEnergyConsumptionReward(RewardFunction):
    """Energy consumption-based reward function.

    Penalises total energy consumption using ``power_breakdown`` from
    the info dict (published by the environment from each infrastructure's
    ``get_electric_consumption``).  This decouples the reward from action
    semantics — only physical power matters.

    Exempts *necessary* charging: when the battery or EV has not yet
    reached its target SoC, that device's consumption is excluded from
    the penalty.  This prevents the reward from conflicting with the
    battery/EV target rewards.

    Normalises by ``max_consumption_kW`` (also from info) so the reward
    stays in [-1, 0].
    """

    def __init__(self, weight: float, name: str = "E_consumption_reward") -> None:
        super().__init__(weight, name)

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        max_step = self.weight * self.max_reward

        if info is None:
            logger.warning("MinimiseEnergyConsumptionReward: info dict is None, returning 0")
            return 0.0, max_step

        penalisable = info.get("penalisable_power_kW")
        max_consumption_kW = info.get("max_consumption_kW")

        if penalisable is None or max_consumption_kW is None:
            logger.warning("MinimiseEnergyConsumptionReward: missing power data in info, returning 0")
            return 0.0, max_step

        if max_consumption_kW <= 0:
            return 0.0, max_step

        # Normalise to [-1, 0]: full consumption = -1, zero consumption = 0
        reward = float(np.clip(-penalisable / max_consumption_kW, -1.0, 0.0))

        return float(self.weight * reward), max_step


# Register MinimiseEnergyConsumption_Reward with the component registry
ComponentRegistry.register('reward', MinimiseEnergyConsumptionReward)
