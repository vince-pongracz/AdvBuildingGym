import logging

import numpy as np
from scipy.optimize import brentq

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


# Solve e^(-x) = x² once at import time.
# This is the zero crossing of f(x) = -x² + e^(-x).
_ZERO_CROSSING_X = brentq(lambda x: -(x ** 2) + np.exp(-x), 0.1, 2.0)


class TempReward(RewardFunction):
    """Temperature comfort reward with gradient across the full error range.

    Uses ``-d² + e^(-d)`` where *d* is a scaled normalised temperature
    error.  The scaling is chosen so that the reward equals zero at
    exactly ``zero_reward_diff_celsius`` degrees of deviation:

    - diff = 0 °C  →  reward = +1.0  (perfect comfort)
    - diff = ``zero_reward_diff_celsius``  →  reward = 0.0
    - diff > ``zero_reward_diff_celsius``  →  reward < 0  (increasingly negative)

    The curve is smooth everywhere, so the agent always has a learning
    signal regardless of distance from setpoint.

    An optional wrong-direction penalty discourages heating when too hot
    (or cooling when too cold), scaled by energy and error magnitude.
    """

    def __init__(
        self,
        weight: float,
        zero_reward_diff_celsius: float,
        wrong_direction_penalty: float,
        terminate_diff_celsius: float,
        terminate_penalty: float,
        name: str = "temp_reward",
    ) -> None:
        """
        Args:
            weight: Global weight applied to the final reward.
            zero_reward_diff_celsius: Temperature difference in °C at which
                the reward is exactly 0.  Errors beyond this yield negative
                rewards.
            wrong_direction_penalty: Penalty coefficient when the HP
                actively works against the temperature error direction.
                Scaled by ``|energy| * |temp_error|``.
        """
        super().__init__(weight, name)
        self.zero_reward_diff_celsius = zero_reward_diff_celsius
        self.wrong_direction_penalty = wrong_direction_penalty
        self.terminate_diff_celsius = terminate_diff_celsius
        self.terminate_penalty = terminate_penalty

    # ------------------------------------------------------------------
    # Termination + reward computation
    # ------------------------------------------------------------------
    @staticmethod
    def _diff_celsius(states) -> float:
        actual_temp = float(states["s_temp_in_norm"][0])
        desired_temp = float(states["s_desired_temp_in_norm"][0])
        diff_norm = abs(actual_temp - desired_temp)
        temp_abs_max = float(states["ctxt_temp_abs_max"][0]) if "ctxt_temp_abs_max" in states else 60.0
        return diff_norm * temp_abs_max

    def should_terminate(self, actions, states, info: dict | None = None) -> bool:
        diff_celsius = self._diff_celsius(states)
        if diff_celsius > self.terminate_diff_celsius:
            logger.info(
                "[%s] terminal step: |T_in - T_set| = %.2f °C > %.2f °C (step %s)",
                self.name, diff_celsius, self.terminate_diff_celsius,
                info.get("iteration") if info else "?",
            )
            return True
        return False

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        actual_temp = float(states["s_temp_in_norm"][0])
        desired_temp = float(states["s_desired_temp_in_norm"][0])
        diff_norm = abs(actual_temp - desired_temp)

        # Convert zero-crossing threshold from °C to normalised space.
        # temp_abs_max is published into the state dict by WeatherDataSource.
        temp_abs_max: float = float(states["ctxt_temp_abs_max"][0]) if "ctxt_temp_abs_max" in states else 60.0

        # Hard band: should_terminate already voted to end the episode in
        # Phase 1; emit the configured terminal penalty here. Skipped when
        # the env disables early termination — the smooth curve below still
        # provides a strong negative signal at large diffs.
        diff_celsius = diff_norm * temp_abs_max
        allow_term = info.get("allow_early_termination", True) if info is not None else True
        if allow_term and diff_celsius > self.terminate_diff_celsius:
            return self.weight * self.terminate_penalty, self.weight * self.max_reward_in_step

        zero_norm = self.zero_reward_diff_celsius / temp_abs_max if temp_abs_max != 0 else 0.0

        # Scale so that the curve crosses zero at exactly zero_norm.
        # _ZERO_CROSSING_X is the x where -x² + e^(-x) = 0.
        scale = _ZERO_CROSSING_X / zero_norm if zero_norm != 0 else 1.0
        d = scale * diff_norm

        # Smooth comfort curve: always provides gradient, no flat regions.
        # d=0 → 1.0;  d=_ZERO_CROSSING_X → 0.0;  d>_ZERO_CROSSING_X → negative.
        reward = float(-(d ** 2) + np.exp(-d))

        # Wrong-direction penalty: heating when too hot, or cooling when too cold
        # HP action: negative = cooling, positive = heating
        if "a_hp" in actions:
            hp_action = float(np.atleast_1d(actions["a_hp"])[0])
            energy = abs(hp_action)
            temp_error = actual_temp - desired_temp  # positive = too hot

            wrong = energy > 0 and (
                (temp_error > 0 and hp_action > 0) or   # too hot but heating
                (temp_error < 0 and hp_action < 0)       # too cold but cooling
            )
            if wrong:
                reward += self.wrong_direction_penalty * energy * abs(temp_error)

        return self.weight * min(reward, 1.0), self.weight * self.max_reward_in_step


# Register TempReward with the component registry
ComponentRegistry.register('reward', TempReward)
