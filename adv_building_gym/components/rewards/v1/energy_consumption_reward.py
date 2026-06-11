import logging

import numpy as np

from adv_building_gym._common.constants import SECONDS_PER_HOUR

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# NOTE VP 2026.01.14. : Add battery life saving reward

class MinimiseEnergyConsumptionReward(RewardFunction):
    """Energy-consumption reward.

    Penalises total consumption via ``power_breakdown`` (info), so only physical
    power matters, not action semantics. Exempts *necessary* charging (battery/EV
    below target SoC) to avoid clashing with the target rewards. Normalised by
    ``max_consumption_kW`` (info) → reward in [-1, 0].
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

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        # TODO VP 2026.06.08.: Implement it
        return 0.0


# register with ComponentRegistry
ComponentRegistry.register('reward', MinimiseEnergyConsumptionReward)
