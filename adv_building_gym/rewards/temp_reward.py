import numpy as np

from .base import RewardFunction


class TempReward(RewardFunction):
    """Temperature-based reward function."""

    def __init__(self, weight: float, name: str = "temp_reward") -> None:
        super().__init__(weight, name)

    def get_reward(self, actions, states) -> float:
        # TODO VP 2026.01.12. : temp diff is regarding the indoor temp and setpoint, check that it is really the case
        temp_diff = states["temp_diff"][0]
        reward = np.exp(-abs(temp_diff))
        return self.weight * reward
