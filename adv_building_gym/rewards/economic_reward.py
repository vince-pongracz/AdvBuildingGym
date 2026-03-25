import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# NOTE VP 2026.01.24. : Maybe switch action positive-negative convention... for now it's okay
# But sometimes it is confusing

# TODO VP 2026.03.25. : Review this reward

class EconomicReward(RewardFunction):
    """Economic-based reward function.

    Action convention: positive consumption (from grid) incurs cost; negative (to grid) earns income.

    Calculates economic reward based on actual power flows in kW and current energy price:
    - Net consumption (positive kW sum) → negative reward (cost)
    - Net production (negative kW sum) → positive reward (income)

    Reads ``net_power_kW`` from the ``info`` dict (published by the environment
    from infrastructure power computations) instead of querying infrastructures
    directly.

    Formula: reward = -net_power_kW × price / (max_power_kW × price_max)
    """

    def __init__(self, weight: float, max_power_kW: float = 25.0, name: str = "economic_reward") -> None:
        """Initialize EconomicReward.

        Args:
            weight: Reward weight for multi-objective optimization.
            max_power_kW: Sum of rated capacities across all infrastructure,
                used as normalisation denominator.
            name: Reward function identifier.
        """
        super().__init__(weight, name)
        self.max_power_kW = max_power_kW

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        max_step = self.weight * self.max_reward

        # E_price is already normalised to [-1, 1], no need to divide by price_max
        current_energy_price = float(states["E_price"][0])

        if info is None:
            logger.warning("EconomicReward: info dict is None, returning 0")
            return 0.0, max_step

        # Read pre-computed net power from info dict (published by environment)
        net_power_kW = info.get("net_power_kW", 0.0)

        if self.max_power_kW <= 0:
            return 0.0, max_step

        # Negative sign: consumption → negative reward (cost); production → positive reward (income)
        reward_economic: float = -net_power_kW * current_energy_price / self.max_power_kW
        return float(self.weight * reward_economic), max_step


# Register EconomicReward with the component registry
ComponentRegistry.register('reward', EconomicReward)
