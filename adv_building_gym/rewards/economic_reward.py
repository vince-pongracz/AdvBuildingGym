import numpy as np

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry


class EconomicReward(RewardFunction):
    """Economic-based reward function.

    Calculates negative of energy cost (cost = price × power consumption).
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

        energy_action = np.array(
            [float(np.atleast_1d(actions[k])[0]) for k in actions.keys() if "_action" in k],
            dtype=np.float32,
        )
        # TODO VP 2025.12.01. : Handle situation with such actions, 
        # which also can bring energy, not just consume...
        energy_action = float(np.sum(energy_action))
        reward_economic: float = current_energy_price * energy_action / energy_price_max
        return self.weight * reward_economic


# Register EconomicReward with the component registry
ComponentRegistry.register('reward', EconomicReward)
