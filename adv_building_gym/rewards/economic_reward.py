import numpy as np

from .base import RewardFunction


class EconomicReward(RewardFunction):
    """Economic-based reward function.

    Calculates negative of energy cost (cost = price × power consumption).
    max_energy_price controls normalisation.
    """

    def __init__(self, weight: float, name: str = "economic_reward") -> None:
        super().__init__(weight, name)

    def get_reward(self, actions, states) -> float:
        energy_price_current = float(states["E_price"][0])
        energy_price_max = float(states["E_price_max"][0])
        # Guard against zero/negative normalisation
        if energy_price_max <= 0:
            energy_price_max = 1.0

        # TODO VP 2025.12.01. : Situation with such actions, which also can bring energy, not just consume...
        action = np.array(
            [float(np.atleast_1d(actions[k])[0]) for k in actions.keys() if "_action" in k],
            dtype=np.float32,
        )
        reward_economic = -energy_price_current * np.abs(action) / energy_price_max
        return float(self.weight * np.sum(reward_economic))
