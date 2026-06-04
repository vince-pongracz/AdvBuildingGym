"""Battery state-of-charge dead-zone reward function (V0)."""

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry


class BatteryTargetRewardV0(RewardFunction):
    """Dead-zone SoC guardrail: 0 inside ``[min_pct, max_pct]``, -1 outside.

    Logic-identical to ``BatteryTargetReward``. Per-step pure-penalty in
    ``[-1, 0]`` with ``max_reward_in_step = 0``.
    """

    def __init__(self,
                weight: float,
                min_pct: float,
                max_pct: float,
                name: str = "battery_target_reward_v0") -> None:
        super().__init__(weight, name)
        if not 0.0 <= min_pct < max_pct <= 1.0:
            raise ValueError("require 0 <= min_pct < max_pct <= 1")

        self.min_pct = float(min_pct)
        self.max_pct = float(max_pct)
        self.max_reward_in_step = 0.0

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        soc = float(states["s_battery_soc"][0])

        if self.min_pct <= soc <= self.max_pct:
            return 0.0, 0.0
        else:
            reward = -1.0

        return self.weight * reward, self.weight * self.max_reward_in_step


ComponentRegistry.register('reward', BatteryTargetRewardV0)
