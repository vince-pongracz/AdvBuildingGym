import logging

import numpy as np

from adv_building_gym.utils.constants import SECONDS_PER_HOUR

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

    def __init__(
        self,
        weight: float,
        threshold_kWh: float,
        name: str = "E_consumption_reward",
    ) -> None:
        super().__init__(weight, name)
        if threshold_kWh < 0:
            raise ValueError("threshold_kWh must be non-negative.")
        self.threshold_kWh = threshold_kWh

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        max_step = self.weight * self.max_reward

        if info is None:
            logger.warning("MinimiseEnergyConsumptionReward: info dict is None, returning 0")
            return 0.0, max_step

        penalisable = info.get("penalisable_power_kW")
        max_consumption_kW = info.get("max_consumption_kW")
        control_step_s = info.get("control_step_s")

        if penalisable is None or max_consumption_kW is None or control_step_s is None:
            logger.warning("MinimiseEnergyConsumptionReward: missing power data in info, returning 0")
            return 0.0, max_step

        if max_consumption_kW <= 0:
            return 0.0, max_step

        # Convert the kWh dead-zone to a kW threshold for this control step:
        # threshold_kW = threshold_kWh / (control_step_s / 3600).
        threshold_kW = self.threshold_kWh * SECONDS_PER_HOUR / float(control_step_s)
        excess_kW = max(0.0, float(penalisable) - threshold_kW)

        # Normalise to [-1, 0]: at/below threshold = 0, full consumption = -1.
        reward = float(np.clip(-excess_kW / max_consumption_kW, -1.0, 0.0))

        return float(self.weight * reward), max_step


# Register MinimiseEnergyConsumption_Reward with the component registry
ComponentRegistry.register('reward', MinimiseEnergyConsumptionReward)
