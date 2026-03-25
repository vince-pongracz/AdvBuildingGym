"""Battery SoC target tracking reward function."""

import numpy as np

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry


# NOTE VP 2026.03.20. : Create a general TargetReward class, which provides some reward modes, but basically rewards getting closer to a target
class BatteryTargetReward(RewardFunction):
    """Reward for keeping the battery state-of-charge close to its target.

    Uses the same exponential-decay pattern as EVChargingReward: full reward
    when |battery_pct - battery_target_pct| is within a small threshold,
    decaying smoothly as the gap grows.
    """

    def __init__(self,
                weight: float,
                name: str = "battery_target_reward",
                diff_threshold: float = 0.02,
                soc_diff_multiplier: float = 5.0) -> None:
        """
        Args:
            weight: Reward weight for multi-objective optimisation.
            name: Reward function identifier.
            diff_threshold: SoC difference below which full reward is given.
                With a 14 kWh pack the default 0.02 corresponds to ~0.28 kWh.
            soc_diff_multiplier: Exponential decay rate.  Higher values
                penalise deviations more sharply.
        """
        super().__init__(weight, name)
        self.diff_threshold = diff_threshold
        self.soc_diff_multiplier = soc_diff_multiplier

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        current_pct = float(states["battery_pct"][0])
        target_pct = float(states["battery_target_pct"][0])
        soc_diff = abs(current_pct - target_pct)

        if soc_diff < self.diff_threshold:
            reward = 1.0
        else:
            reward = float(np.exp(-soc_diff * self.soc_diff_multiplier))

        return self.weight * reward, self.weight * self.max_reward


ComponentRegistry.register('reward', BatteryTargetReward)
