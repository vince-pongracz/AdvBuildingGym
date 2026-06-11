import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryMgmtReward(RewardFunction):
    """Terminal penalty for ending the episode below the starting SoC.

    Asymmetric: ending below start is penalised, ending equal/above gives 0 — preserves
    the battery for next episode without fighting EconomicReward's mid-episode cycling.
    Terminal step: deficit = max(0, start_soc - end_soc); reward = -deficit/scale.
    """

    def __init__(self,
                weight: float,
                scale: float = 0.5,
                name: str = "battery_mgmt_reward") -> None:
        super().__init__(weight, name)
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        self.scale: float = scale
        self.episode_start_soc: float = 0.0

    def on_reset(self, states, info: dict | None = None) -> None:
        self.episode_start_soc = float(states["s_battery_soc"][0])

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        if info is None or not info.get("terminated", False):
            return 0.0

        episode_end_soc = float(next_state["s_battery_soc"][0])  # resulting SoC (s')
        deficit = max(0.0, self.episode_start_soc - episode_end_soc)
        reward = -1.0 * float(deficit / self.scale)
        return self.weight * reward


ComponentRegistry.register('reward', BatteryMgmtReward)
