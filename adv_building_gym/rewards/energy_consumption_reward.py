import numpy as np

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry


# TODO VP 2026.01.14. : Add battery life saving reward

class MinimiseEnergyConsumption_Reward(RewardFunction):
    """Energy consumption-based reward function."""

    def __init__(self, weight: float, name: str = "E_consumption_reward") -> None:
        super().__init__(weight, name)

    def get_reward(self, actions, states) -> float:
        e_consumption: float = 0
        for key, v in actions.items():
            # HP_action is 2D: [energy, mode], only use energy (index 0)
            if key == "HP_action":
                e_consumption += float(np.atleast_1d(v)[0])
            else:
                # Other actions are 1D, sum as before
                e_consumption += np.sum(v, axis=0)

        e_consumption_max = len(actions)
        reward = e_consumption / e_consumption_max

        return float(self.weight * reward)


# Register MinimiseEnergyConsumption_Reward with the component registry
ComponentRegistry.register('reward', MinimiseEnergyConsumption_Reward)
