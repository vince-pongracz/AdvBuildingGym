import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryMgmtRewardV0(RewardFunction):
    """Terminal SoC-deficit reward (V0), sparse, scaled to ``episode_length``.

    Fires only on the terminal step (``info["terminated"]``). Asymmetric: ending below
    the starting SoC is penalised, ending equal/above gives 0. Range ``[-episode_length, 0]``:
    deficit = max(0, start_soc - end_soc); reward = -clip(deficit/scale, 0, 1) * episode_length.
    ``scale`` sets how fast the deficit saturates the penalty (smaller = sooner).
    """

    def __init__(self,
                weight: float,
                scale: float = 1.0,
                name: str = "battery_mgmt_reward_v0") -> None:
        super().__init__(weight, name)
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        self.scale: float = float(scale)
        self.episode_start_soc: float = 0.0

    def on_reset(self, states, info: dict | None = None) -> None:
        self.episode_start_soc = float(states["s_battery_soc"][0])

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        if info is None or not info.get("terminated", False):
            return 0.0

        episode_length = float(info.get("episode_length", 1))
        episode_end_soc = float(next_state["s_battery_soc"][0])  # resulting SoC (s')
        deficit = max(0.0, self.episode_start_soc - episode_end_soc)
        reward = -float(np.clip(deficit / self.scale, 0.0, 1.0)) * episode_length
        return float(self.weight * reward)


ComponentRegistry.register('reward', BatteryMgmtRewardV0)
