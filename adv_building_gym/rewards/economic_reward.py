import numpy as np

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry

# NOTE VP 2026.01.24. : Maybe switch action positive-negative convention... for now it's okay
# But sometimes it is confusing

class EconomicReward(RewardFunction):
    """Economic-based reward function.

    Action convention: positive = consumption (from grid), negative = production (to grid).

    Calculates economic reward based on net energy flow and price:
    - Net consumption (positive action sum) → negative reward (cost to buy from grid)
    - Net production (negative action sum) → positive reward (income from selling to grid)

    Formula: reward = -price × Σ(actions) / max_price

    max_energy_price controls normalisation.
    """

    def __init__(self, weight: float, name: str = "economic_reward") -> None:
        super().__init__(weight, name)

    # NOTE VP 2026.01.23. : Option for a TODO to realise different price for selling and buying electricity
    def get_reward(self, actions, states) -> float:
        current_energy_price = float(states["E_price"][0])
        energy_price_max = float(states["E_price_max"][0])
        # Guard against zero/negative normalisation
        if energy_price_max <= 0:
            energy_price_max = 1.0

        # Sum all actions to get net energy flow
        # Positive sum = net consumption from grid (should be penalized)
        # Negative sum = net production to grid (should be rewarded)
        energy_action = np.array(
            [float(np.atleast_1d(actions[k])[0]) for k in actions.keys() if "_action" in k],
            dtype=np.float32,
        )
        net_energy = float(np.sum(energy_action))

        # Negative sign ensures: consumption gives negative reward (cost), production gives positive reward (income)
        reward_economic: float = -current_energy_price * net_energy / energy_price_max
        return self.weight * reward_economic


# Register EconomicReward with the component registry
ComponentRegistry.register('reward', EconomicReward)
