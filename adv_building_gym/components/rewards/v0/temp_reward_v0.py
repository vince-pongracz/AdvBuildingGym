import logging

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import TEMP_ABS_MAX_CELSIUS

logger = logging.getLogger(__name__)


class TempRewardV0(RewardFunction):
    """Bounded temperature comfort reward (V0), range (-1, 1].

    ``d = |T_in - T_set|`` °C (from the normalised error, scaled by the fixed
    ``info["temp_abs_max"]``). Peaks +1 at d=0 (precision driver ``2*exp(-|1.4 d|)-1``)
    and decreases towards -1 as d grows.
    """

    def __init__(self, weight: float, name: str = "temp_reward_v0") -> None:
        super().__init__(weight, name)
        self.x_scale:float = 1.4
        self.exp_scale:float = 2.0

    @staticmethod
    def _diff_celsius(next_state, info) -> float:
        # Signed comfort error (indoor temp − setpoint) at s', from the single
        # s_temp_error_norm observation published by InsideTemperature. The fixed
        # temperature scale is read from the info channel (WeatherDataSource).
        error_norm = float(next_state["s_temp_error_norm"][0])
        temp_abs_max = float(info["temp_abs_max"]) if info is not None and "temp_abs_max" in info else TEMP_ABS_MAX_CELSIUS
        return error_norm * temp_abs_max

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        d_celsius = self._diff_celsius(next_state, info)
        
        reward_precision_driver = self.exp_scale * float(np.exp(-abs(self.x_scale * d_celsius))) - 1.0
        x_nullpoint_pos = -1.0 / self.x_scale * np.log(1.0 / self.exp_scale)
        reward_precision_driver = np.clip(reward_precision_driver, 0.0, 1.0)
        
        reward_slow_driver = 0.0
        THRESHOLD:float = 3.0
        THRESHOLD_SLOW_DRIVER = THRESHOLD - x_nullpoint_pos  # slow driver ~-1 at 3°C diff, 0 at 0°C
        if d_celsius > 0.0:
            reward_slow_driver = -(d_celsius - x_nullpoint_pos) / THRESHOLD_SLOW_DRIVER
        else:
            reward_slow_driver = (d_celsius + x_nullpoint_pos) / THRESHOLD_SLOW_DRIVER
        reward_slow_driver = np.clip(reward_slow_driver, -1.0, 0.0)

        # TODO VP 2026.06.10.: Check the reward_diagnostics, how does it appear in the TB logs.
        # Per-step diagnostics (info["reward_diagnostics"]): where |d_celsius| sits vs the slow driver.
        #  - violated: |d| > THRESHOLD (slow driver saturated at -1).
        #  - in_area:  x_nullpoint_pos <= |d| <= THRESHOLD (slow driver's active band).
        # EpisodeMetricsCallback sums per episode; EvalStateActionCallback emits cumulative per round.
        self._publish_diagnostics(
            info,
            violated=abs(d_celsius) > THRESHOLD,
            in_area=x_nullpoint_pos <= abs(d_celsius) <= THRESHOLD,
        )

        reward = reward_precision_driver + reward_slow_driver
        reward = float(np.clip(reward, -1.0, 1.0))
        return self.weight * reward

    def _publish_diagnostics(self, info: dict | None, *, violated: bool, in_area: bool) -> None:
        """Record per-step 0/1 flags into ``info["reward_diagnostics"]`` (cleared each pass).

        Summed per episode by EpisodeMetricsCallback; cumulative per round by EvalStateActionCallback.
        """
        if info is None:
            return
        diag = info.setdefault("reward_diagnostics", {})
        diag[f"{self.name}/slow_driver_violated"] = float(violated)
        diag[f"{self.name}/slow_driver_in_area"] = float(in_area)


ComponentRegistry.register('reward', TempRewardV0)
