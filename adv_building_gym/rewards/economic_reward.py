import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# NOTE VP 2026.01.24. : Maybe switch action positive-negative convention... for now it's okay
# But sometimes it is confusing

class EconomicReward(RewardFunction):
    """Economic-based reward function.

    Action convention: positive consumption (from grid) incurs cost; negative (to grid) earns income.

    Calculates economic reward based on actual power flows in kW and current energy price:
    - Net consumption (positive kW sum) → negative reward (cost)
    - Net production (negative kW sum) → positive reward (income)

    Reads ``net_power_kW``, ``max_consumption_kW``, and ``max_export_kW``
    from the ``info`` dict (published by the environment from infrastructure
    power computations) instead of querying infrastructures directly.

    Normalisation uses the directional bound that matches the sign of
    ``net_power_kW``: ``max_consumption_kW`` when positive (grid draw),
    ``max_export_kW`` when negative (grid feed-in). This keeps the reward
    symmetric in [-1, 1] despite asymmetric power bounds.

    ``E_price`` is already normalised to [-1, 1] so no additional price
    scaling is needed.
    """

    def __init__(self, weight: float, name: str = "economic_reward") -> None:
        """Initialize EconomicReward.

        Args:
            weight: Reward weight for multi-objective optimization.
            name: Reward function identifier.
        """
        super().__init__(weight, name)

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        max_step = self.weight * self.max_reward

        # E_price is already normalised to [-1, 1], no need to divide by price_max
        current_energy_price = float(states["E_price"][0])

        if info is None:
            logger.warning("EconomicReward: info dict is None, returning 0")
            return 0.0, max_step

        # Read pre-computed values from info dict (published by environment)
        net_power_kW = info.get("net_power_kW")
        max_consumption_kW = info.get("max_consumption_kW")
        max_export_kW = info.get("max_export_kW")

        if net_power_kW is None or max_consumption_kW is None or max_export_kW is None:
            logger.warning("EconomicReward: missing power data in info, returning 0")
            return 0.0, max_step

        # Pick the normalisation bound matching the power flow direction
        if net_power_kW >= 0:
            denominator = max_consumption_kW
        else:
            denominator = max_export_kW

        if denominator <= 0:
            return 0.0, max_step

        # Negative sign: consumption → negative reward (cost); production → positive reward (income)
        reward_economic = float(np.clip(
            -net_power_kW * current_energy_price / denominator, -1.0, 1.0
        ))
        return float(self.weight * reward_economic), max_step


# Register EconomicReward with the component registry
ComponentRegistry.register('reward', EconomicReward)
