import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryMgmtReward(RewardFunction):
    """Terminal reward penalising end-of-episode SoC deficit vs. start.

    Asymmetric by design: only ending *below* the starting SoC is
    penalised; ending equal or above yields 0. This preserves the
    battery for the next episode (or next day) without fighting an
    EconomicReward that needs to cycle SoC for arbitrage — the agent
    is free to drain mid-episode as long as it replenishes by the end.

    Shape (only on the terminal step):
        deficit = max(0, episode_start_soc - episode_end_soc)
        reward  = -clip(deficit / scale, 0, 1)        in [-1, 0]
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
        self.episode_start_soc = float(states["s_battery_pct"][0])

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        if info is None or not info.get("terminated", False):
            return 0.0, 0.0

        episode_end_soc = float(states["s_battery_pct"][0])
        deficit = max(0.0, self.episode_start_soc - episode_end_soc)
        reward = -1.0 * float(np.clip(deficit / self.scale, 0.0, 1.0))
        return self.weight * reward, self.weight * self.max_reward_in_step


ComponentRegistry.register('reward', BatteryMgmtReward)
