import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryMgmtRewardV0(RewardFunction):
    """Terminal SoC-deficit reward (V0), sparse, scaled to ``episode_length``.

    Fires only on the env's terminal step (``info["terminated"]``), which
    happens at the natural end of the episode. Asymmetric by design:
    only ending *below* the starting SoC is penalised; ending equal or
    above yields 0.

    Per-fire range: ``[-episode_length, 0]``.

        deficit = max(0, episode_start_soc - episode_end_soc)   # in [0, 1]
        reward  = -clip(deficit / scale, 0, 1) * episode_length

    ``scale`` controls how big a SoC deficit saturates the penalty
    (default ``1.0`` → linear in ``deficit``; smaller ``scale`` saturates
    sooner).
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
        self.episode_start_soc = float(states["s_battery_pct"][0])

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        if info is None or not info.get("terminated", False):
            return 0.0, 0.0

        episode_length = float(info.get("episode_length", 1))
        episode_end_soc = float(states["s_battery_pct"][0])
        deficit = max(0.0, self.episode_start_soc - episode_end_soc)
        reward = -float(np.clip(deficit / self.scale, 0.0, 1.0)) * episode_length
        return float(self.weight * reward), float(self.weight * episode_length)


ComponentRegistry.register('reward', BatteryMgmtRewardV0)
