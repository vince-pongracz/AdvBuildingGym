import numpy as np
from typing import ClassVar, List, Set

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry

# NOTE VP 2026.01.24. : Maybe switch action positive-negative convention... for now it's okay
# But sometimes it is confusing

class EconomicReward(RewardFunction):
    """Economic-based reward function.

    Action convention: positive consumption (from grid) incurs cost; negative (to grid) earns income.

    Calculates economic reward based on actual power flows in kW and current energy price:
    - Net consumption (positive kW sum) → negative reward (cost)
    - Net production (negative kW sum) → positive reward (income)

    Uses get_electric_consumption() on each infrastructure to obtain real kW values,
    so devices with different rated powers are weighted correctly.

    Formula: reward = -net_power_kW × price / (max_power_kW × price_max)
    """

    # infrastructures comes from context (the Config's infras list) — excluded from serialization
    _context_params: ClassVar[Set[str]] = {'infrastructures'}

    def __init__(self, infrastructures: List, weight: float, name: str = "economic_reward") -> None:
        super().__init__(weight, name)
        self.infrastructures = infrastructures
        # Sum of rated capacities used as normalisation denominator
        self.max_power_kW = sum(infra.Q_electric_max for infra in infrastructures)

    def get_reward(self, actions, states) -> float:
        current_energy_price = float(states["E_price"][0])
        energy_price_max = float(states["E_price_max"][0])
        if energy_price_max <= 0:
            energy_price_max = 1.0

        # Sum actual kW across all infrastructure (positive = grid import, negative = export)
        net_power_kW = sum(
            infra.get_electric_consumption(actions) for infra in self.infrastructures
        )

        if self.max_power_kW <= 0:
            return 0.0

        # Negative sign: consumption → negative reward (cost); production → positive reward (income)
        reward_economic: float = -net_power_kW * current_energy_price / (
            self.max_power_kW * energy_price_max
        )
        return float(self.weight * reward_economic)


# Register EconomicReward with the component registry
ComponentRegistry.register('reward', EconomicReward)
