"""EV charging SoC tracking reward function."""

import numpy as np

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry


class EVChargingReward(RewardFunction):
    """Reward for keeping the EV state-of-charge close to its target.

    Uses the same exponential-decay pattern as TempReward: full reward
    when |ev_soc - ev_target_soc| is within a small threshold, decaying
    smoothly as the gap grows.

    When the EV is not connected the objective is trivially satisfied
    and the reward is 1.0 (neutral).
    """

    def __init__(self,
                 weight: float,
                 name: str = "ev_charging_reward",
                 diff_threshold: float = 0.02,
                 soc_diff_multiplier: float = 5.0) -> None:
        """
        Args:
            weight: Reward weight for multi-objective optimisation.
            name: Reward function identifier.
            diff_threshold: SoC difference below which full reward is given.
                With a 60 kWh battery the default 0.02 corresponds to ~1.2 kWh.
            soc_diff_multiplier: Exponential decay rate.  Higher values
                penalise deviations more sharply.
        """
        super().__init__(weight, name)
        self.diff_threshold = diff_threshold
        self.soc_diff_multiplier = soc_diff_multiplier

    def get_reward(self, _actions, states) -> float:
        ev_connected = float(states["ev_connected"][0])

        if ev_connected < 0.5:
            return self.weight * 1.0

        current_soc = float(states["ev_soc"][0])
        target_soc = float(states["ev_target_soc"][0])
        soc_diff = abs(current_soc - target_soc)

        if soc_diff < self.diff_threshold:
            reward = 1.0
        else:
            reward = float(np.exp(-soc_diff * self.soc_diff_multiplier))

        return self.weight * reward


ComponentRegistry.register('reward', EVChargingReward)
