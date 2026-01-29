import numpy as np

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry


class TempReward(RewardFunction):
    """
    Temperature-based reward function.
    This reward encourages the agent to maintain the indoor temperature
    close to the desired user temperature setpoint.
    """

    def __init__(self, weight: float, 
                 name: str = "temp_reward",
                 diff_threshold: float = 0.3) -> None:
        """
        Initialize TempReward.
        Args:
            diff_threshold: Temperature difference threshold for full reward
        """
        
        super().__init__(weight, name)
        self.diff_threshold = diff_threshold

    def get_reward(self, actions, states) -> float:
        """
        Calculate temperature comfort reward based on difference between
        actual and desired indoor temperature.

        Returns exponential reward that approaches 1 when temperatures match
        and decreases as the difference increases.
        """
        actual_temp = states["temp_norm_in"][0]
        desired_temp = states["desired_temp_in_norm"][0]
        temp_diff = abs(actual_temp - desired_temp)
        
        if temp_diff < self.diff_threshold:
            reward = 1.0
        else:
            reward = np.exp(-temp_diff)

        return self.weight * reward


# Register TempReward with the component registry
ComponentRegistry.register('reward', TempReward)
