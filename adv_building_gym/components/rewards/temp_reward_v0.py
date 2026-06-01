import logging

import numpy as np

from .base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class TempRewardV0(RewardFunction):
    """Bounded temperature comfort reward (V0).

    Shape: ``2 * exp(-|1.4 * d|) - 1`` where ``d = |T_in - T_set|`` in °C
    (recovered from the normalised state via ``ctxt_temp_abs_max``,
    falling back to 60 °C) -- if we are close to setpoint.

    Range: ``(-1, 1]`` — peaks at ``+1`` when ``d = 0`` and decreases linear towards
    ``-1`` as ``d`` increases, reaching -1 at around 40 °C diff.
    """

    def __init__(self, weight: float, name: str = "temp_reward_v0") -> None:
        super().__init__(weight, name)
        self.x_scale:float = 1.4
        self.exp_scale:float = 2.0

    @staticmethod
    def _diff_celsius(states) -> float:
        actual_temp = float(states["s_temp_in_norm"][0])
        desired_temp = float(states["s_desired_temp_in_norm"][0])
        diff_norm = actual_temp - desired_temp
        temp_abs_max = float(states["ctxt_temp_abs_max"][0]) if "ctxt_temp_abs_max" in states else 60.0
        return diff_norm * temp_abs_max

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        d_celsius = self._diff_celsius(states)
        
        reward_precision_driver = self.exp_scale * float(np.exp(-abs(self.x_scale * d_celsius))) - 1.0
        x_nullpoint_pos = -1.0 / self.x_scale * np.log(1.0 / self.exp_scale)
        reward_precision_driver = np.clip(reward_precision_driver, 0.0, 1.0)
        
        reward_slow_driver = 0.0
        THRESHOLD:float = 3.0
        THRESHOLD_SLOW_DRIVER = THRESHOLD - x_nullpoint_pos # At around 3 °C diff, the reward for slow driver reaches -1.0, and is 0 at 0 °C diff.
        if d_celsius > 0.0:
            reward_slow_driver = -(d_celsius - x_nullpoint_pos) / THRESHOLD_SLOW_DRIVER
        else:
            reward_slow_driver = (d_celsius + x_nullpoint_pos) / THRESHOLD_SLOW_DRIVER
        reward_slow_driver = np.clip(reward_slow_driver, -1.0, 0.0)

        # Per-step diagnostics for TensorBoard (via info["reward_diagnostics"]).
        # Two flags describe where |d_celsius| sits relative to the slow driver:
        #  - violated: |d_celsius| > THRESHOLD, i.e. the slow driver has
        #    saturated at -1 and no longer provides comfort gradient.
        #  - in_area:  x_nullpoint_pos <= |d_celsius| <= THRESHOLD, i.e. inside
        #    the slow driver's active band (between the precision plateau,
        #    where it clips to 0, and saturation). |d| < x_nullpoint_pos is
        #    neither flag (the precision driver dominates there).
        # EpisodeMetricsCallback sums each per episode (a total count);
        # EvalStateActionCallback emits the running cumulative sum per eval
        # round so the curve grows by 1 at each step where the flag fires.
        self._publish_diagnostics(
            info,
            violated=abs(d_celsius) > THRESHOLD,
            in_area=x_nullpoint_pos <= abs(d_celsius) <= THRESHOLD,
        )

        reward = reward_precision_driver + reward_slow_driver
        reward = float(np.clip(reward, -1.0, 1.0))
        return self.weight * reward, self.weight * self.max_reward_in_step

    def _publish_diagnostics(self, info: dict | None, *, violated: bool, in_area: bool) -> None:
        """Record per-step 0/1 flags into the shared reward-diagnostics channel.

        The env clears ``info["reward_diagnostics"]`` before each reward pass.
        EpisodeMetricsCallback sums these per key across the episode (total
        count); EvalStateActionCallback turns them into per-step cumulative
        sums for the eval-round trajectory charts.
        """
        if info is None:
            return
        diag = info.setdefault("reward_diagnostics", {})
        diag[f"{self.name}/slow_driver_violated"] = float(violated)
        diag[f"{self.name}/slow_driver_in_area"] = float(in_area)


ComponentRegistry.register('reward', TempRewardV0)
