import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryRangeMgmtRewardV0(RewardFunction):
    """Terminal SoC-deficit reward (V0), sparse, scaled to ``episode_length``.

    Fires only on the terminal step (``info["terminated"]``). 
    Ending not in the vinicity of the starting SoC is penalised, ending ``allowed_deviation`` close to it gives 0. 
    Range ``[-episode_length, 0]``
    """

    def __init__(self,
                weight: float,
                allowed_deviation: float = 0.05,
                name: str = "battery_range_mgmt_reward_v0") -> None:
        super().__init__(weight, name)
        self.episode_start_soc: float = 0.0
        self.allowed_deviation: float = allowed_deviation

    def on_reset(self, states, info: dict) -> None:
        self.episode_start_soc = float(states["s_battery_soc"][0])

    def get_reward(self, actions, state, next_state, info: dict) -> float:
        if not info.get("terminated", False):
            return 0.0

        episode_length = float(info.get("episode_length", 1))
        episode_end_soc = float(next_state["s_battery_soc"][0])  # resulting SoC (s')
        is_soc_deficit = np.abs(self.episode_start_soc - episode_end_soc) > self.allowed_deviation
        if is_soc_deficit:
            reward = -1.0 * episode_length
        else:
            reward = 0.0
        return float(self.weight * reward)


ComponentRegistry.register('reward', BatteryRangeMgmtRewardV0)
