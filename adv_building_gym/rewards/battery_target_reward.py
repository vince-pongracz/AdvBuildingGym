"""Battery state-of-charge dead-zone reward function."""

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

class BatteryTargetReward(RewardFunction):
    """Dead-zone SoC guardrail: 0 inside ``[min_pct, max_pct]``, -1 outside.

    Reads ``s_battery_pct`` from the observation dict. The band is owned
    by this reward (constructor args); the battery component is unaware.
    """

    def __init__(self,
                weight: float,
                min_pct: float,
                max_pct: float,
                name: str = "battery_target_reward") -> None:
        """
        Args:
            weight: Reward weight for multi-objective optimisation.
            name: Reward function identifier.
            min_pct: Lower edge of the healthy SoC band (fractional SoC).
            max_pct: Upper edge of the healthy SoC band (fractional SoC).
        """
        super().__init__(weight, name)
        if not 0.0 <= min_pct < max_pct <= 1.0:
            raise ValueError("require 0 <= min_pct < max_pct <= 1")

        self.min_pct = float(min_pct)
        self.max_pct = float(max_pct)

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        soc = float(states["s_battery_pct"][0])

        if self.min_pct <= soc <= self.max_pct:
            return 0.0, 0.0
        else:
            reward = -1.0

        return self.weight * reward, self.weight * self.max_reward_in_step


ComponentRegistry.register('reward', BatteryTargetReward)
