import logging

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class TempRewardV0(RewardFunction):
    """Bounded temperature comfort reward (V0), range (-1, 1].

    ``d = |T_in - T_set|`` °C (from normalised state via ``ctxt_temp_abs_max``, default 60).
    Peaks +1 at d=0 (precision driver ``2*exp(-|1.4 d|)-1``) and decreases towards -1 as d grows.
    """

    def __init__(self, weight: float, name: str = "temp_reward_v0") -> None:
        super().__init__(weight, name)
        self.x_scale:float = 1.4
        self.exp_scale:float = 2.0

    @staticmethod
    def _diff_celsius(state, next_state) -> float:
        # Resulting indoor temperature (s') vs the setpoint the agent observed (s).
        actual_temp = float(next_state["s_temp_in_norm"][0])
        desired_temp = float(state["s_desired_temp_in_norm"][0])
        diff_norm = actual_temp - desired_temp
        temp_abs_max = float(next_state["ctxt_temp_abs_max"][0]) if "ctxt_temp_abs_max" in next_state else 60.0
        return diff_norm * temp_abs_max

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        d_celsius = self._diff_celsius(state, next_state)
        
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
