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
                 diff_threshold: float = 0.02,
                 wrong_direction_penalty: float = 0.0) -> None:
        """
        Initialize TempReward.
        Args:
            diff_threshold: Temperature difference threshold for full reward
            wrong_direction_penalty: Penalty applied when the HP mode works
                against the temperature error direction (e.g. heating when
                already too hot). Subtracted from the comfort reward.
        """

        super().__init__(weight, name)
        self.diff_threshold = diff_threshold
        self.temp_const_multiplier = 0.25
        self.wrong_direction_penalty = wrong_direction_penalty

    def get_reward(self, actions, states) -> float:
        """
        Calculate temperature comfort reward based on difference between
        actual and desired indoor temperature.

        Returns exponential reward that approaches 1 when temperatures match
        and decreases as the difference increases.
        """
        actual_temp = float(states["temp_in_norm"][0])
        desired_temp = float(states["desired_temp_in_norm"][0])

        diff_1: float = abs(1.0 - (actual_temp / desired_temp))
        diff_2: float = abs(1.0 - (desired_temp / actual_temp))
        temp_diff = np.mean([diff_1, diff_2])

        if temp_diff < self.diff_threshold:
            reward = 1.0
        else:
            reward = np.exp(-temp_diff * self.temp_const_multiplier)

        # Penalise wrong HP mode direction: heating when too hot or cooling
        # when too cold works against the comfort objective and wastes energy.
        # HP mode convention: <0.4 = cooling, >0.6 = heating, [0.4, 0.6] = off
        if "HP_action" in actions:
            energy = float(np.atleast_1d(actions["HP_action"])[0])
            mode = float(np.atleast_1d(actions["HP_action"])[1])
            temp_error = actual_temp - desired_temp  # positive = too hot

            if energy > 0:
                if temp_error > 0 and mode > 0.6:
                    # Heating when already too hot
                    reward = self.wrong_direction_penalty
                elif temp_error < 0 and mode < 0.4:
                    # Cooling when already too cold
                    reward = self.wrong_direction_penalty

        return self.weight * reward


# Register TempReward with the component registry
ComponentRegistry.register('reward', TempReward)
