import logging

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import TEMP_ABS_MAX_CELSIUS

logger = logging.getLogger(__name__)


class TempReward(RewardFunction):
    """Temperature comfort reward with a gradient across the full error range.

    Let ``d = |T_in - T_set|`` °C (from the normalised error, scaled by the fixed
    ``info["temp_abs_max"]``). Shaped reward, continuous at ``d = zero_reward_diff_celsius`` (d0):
    - ``d < d0``: concave parabola ``1 - (d / d0)²`` — peaks at +1.0 when
      ``d = 0`` and crosses zero at ``d = d0``.
    - ``d ≥ d0``: linear tail ``-(d - d0)`` — slope -1 °C⁻¹.

    Optional wrong-direction penalty when the HP heats while too hot (or cools while
    too cold), scaled by ``|a_hp| * min(|°C error|, floor_diff_celsius)``.

    Hard band via ``should_terminate``: ``d > terminate_diff_celsius`` ends the episode
    (unless ``info["allow_early_termination"] = False``) and returns ``terminate_penalty``.
    Final reward (curve + wrong-direction) is capped at +1 before ``weight``.
    """

    def __init__(
        self,
        weight: float,
        zero_reward_diff_celsius: float,
        wrong_direction_penalty: float,
        terminate_diff_celsius: float,
        terminate_penalty: float,
        floor_diff_celsius: float = 30.0,
        name: str = "temp_reward",
    ) -> None:
        """
        Args:
            weight: Global weight applied to the final reward.
            zero_reward_diff_celsius: °C deviation at which the shaped
                curve crosses zero. Inside this band the reward is a
                concave parabola peaking at +1.0; beyond it the reward
                is linear with slope -1 °C⁻¹.
            wrong_direction_penalty: Coefficient added when the HP works
                against the temperature error. Scaled by
                ``|a_hp| * min(|°C error|, floor_diff_celsius)``.
            terminate_diff_celsius: Hard comfort band. ``should_terminate``
                ends the episode when ``|T_in - T_set|`` exceeds this.
            terminate_penalty: Single-step reward emitted when the hard
                band is breached (subject to ``allow_early_termination``).
            floor_diff_celsius: °C error at which the linear tail and the
                wrong-direction scaler stop growing. Caps the magnitude
                of penalties to keep SAC gradients stable when early
                termination is disabled.
            name: Identifier used in logging and the per-component reward
                breakdown.
        """
        super().__init__(weight, name)
        self.zero_reward_diff_celsius = zero_reward_diff_celsius
        self.k = 1 / zero_reward_diff_celsius ** 2
        self.wrong_direction_penalty = wrong_direction_penalty
        self.terminate_diff_celsius = terminate_diff_celsius
        self.terminate_penalty = terminate_penalty
        self.floor_diff_celsius = floor_diff_celsius

    # ------------------------------------------------------------------
    # Termination + reward computation
    # ------------------------------------------------------------------
    @staticmethod
    def _temp_error_celsius(next_state, info) -> float:
        # Signed comfort error (indoor temp − setpoint) in °C at the resulting state s'
        # (+ve = too hot). s_temp_error_norm is the single normalised error published by
        # InsideTemperature; the fixed temperature scale comes from the info channel
        # (WeatherDataSource), falling back to the module constant when absent.
        error_norm = float(next_state["s_temp_error_norm"][0])
        temp_abs_max = float(info["temp_abs_max"]) if "temp_abs_max" in info else TEMP_ABS_MAX_CELSIUS
        return error_norm * temp_abs_max

    def should_terminate(self, actions, state, next_state, info: dict) -> bool:
        diff_celsius = abs(self._temp_error_celsius(next_state, info))
        if diff_celsius > self.terminate_diff_celsius:
            logger.info(
                "[%s] terminal step: |T_in - T_set| = %.2f °C > %.2f °C (step %s)",
                self.name, diff_celsius, self.terminate_diff_celsius,
                info.get("iteration", "?"),
            )
            return True
        return False

    def get_reward(self, actions, state, next_state, info: dict) -> float:
        # Signed comfort error in °C (+ve = too hot); the absolute value drives the curve.
        temp_error_celsius = self._temp_error_celsius(next_state, info)
        diff_celsius = abs(temp_error_celsius)

        # Hard band: should_terminate already voted to end; emit terminal penalty.
        # Skipped when early termination is disabled (the curve still signals strongly).
        allow_term = info.get("allow_early_termination", True)
        if allow_term and diff_celsius > self.terminate_diff_celsius:
            return self.weight * self.terminate_penalty

        # Clamp °C error to floor_diff_celsius so the negative tail stays bounded
        # (large drift would otherwise spike the penalty and destabilise SAC).
        diff_celsius = min(diff_celsius, self.floor_diff_celsius)

        # Smooth comfort curve: concave parabola peaking at +1 inside the zero-reward
        # band, then a linear tail with slope -1 °C⁻¹ beyond it.
        if diff_celsius < self.zero_reward_diff_celsius:
            reward = 1.0 - self.k * diff_celsius ** 2
        else:
            reward = self.zero_reward_diff_celsius - diff_celsius

        # Wrong-direction penalty: heating when too hot / cooling when too cold
        # (a_hp: negative=cool, positive=heat). Scaled by |°C error| (clamped) for
        # a stable meaning across temp_abs_max.
        if "a_hp" in actions:
            hp_action = float(np.atleast_1d(actions["a_hp"])[0])
            energy = abs(hp_action)
            wrong = energy > 0 and (
                (temp_error_celsius > 0 and hp_action > 0) or   # too hot but heating
                (temp_error_celsius < 0 and hp_action < 0)      # too cold but cooling
            )
            if wrong:
                error_magnitude_celsius = min(abs(temp_error_celsius), self.floor_diff_celsius)
                reward += self.wrong_direction_penalty * energy * error_magnitude_celsius

        return self.weight * min(reward, 1.0)


# register with ComponentRegistry
ComponentRegistry.register('reward', TempReward)
