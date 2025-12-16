
from abc import ABC

import numpy as np


class RewardFunction(ABC):
    def __init__(self, weight: float, name: str = "default") -> None:
        self.weight = weight
        self.name = name

    def get_reward(self, actions, states) -> float:
        raise NotImplementedError()

class TempReward(RewardFunction):
    def __init__(self, weight: float, name: str = "temp_reward") -> None:
        super().__init__(weight, name)

    def get_reward(self, actions, states) -> float:
        temp_diff = states["temp_diff"][0]
        reward = np.exp(-abs(temp_diff))
        return self.weight * reward

# Economic-based reward: negative of energy cost (cost = price × power consumption)
# max_energy_price: controls normalisation
class EconomicReward(RewardFunction):
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

class EnergyConsumptionReward(RewardFunction):
    def __init__(self, weight: float, name: str = "E_consumption_reward") -> None:
        super().__init__(weight, name)

    def get_reward(self, actions, states) -> float:
        e_consumption: float = 0
        for _, v in actions.items():
            e_consumption += np.sum(v, axis=0)

        e_consumption_max = len(actions)
        reward = e_consumption / e_consumption_max

        return float(self.weight * reward)


# Something with network stability?
def flexibility_reward() -> float:
    return 0
