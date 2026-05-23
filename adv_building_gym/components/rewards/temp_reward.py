import logging

import numpy as np

from .base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class TempReward(RewardFunction):
    """Temperature comfort reward with a gradient across the full error range.

    Let ``d = |T_in - T_set|`` in °C (recovered from the normalised state
    via ``ctxt_temp_abs_max``, falling back to 60 °C). The shaped reward
    is piecewise, continuous at ``d = zero_reward_diff_celsius`` (``d0``):

    - ``d < d0``: concave parabola ``1 - (d / d0)²`` — peaks at +1.0 when
      ``d = 0`` and crosses zero at ``d = d0``.
    - ``d ≥ d0``: linear tail ``-(d - d0)`` — slope -1 °C⁻¹.

    Before the curve is evaluated ``d`` is clamped to
    ``floor_diff_celsius`` so the linear tail cannot drive arbitrarily
    large negative rewards (matters mainly when early termination is
    disabled — otherwise the hard band below fires first).

    An optional wrong-direction penalty fires when the HP heats while the
    room is too hot (or cools while too cold), scaled by
    ``|a_hp| * min(|°C error|, floor_diff_celsius)``. Anchoring on °C
    keeps the coefficient physically meaningful across different
    ``temp_abs_max`` values.

    A hard comfort band is enforced via ``should_terminate``: when
    ``d > terminate_diff_celsius`` the episode is ended (unless the env
    sets ``info["allow_early_termination"] = False``) and the step
    returns ``terminate_penalty`` instead of the shaped curve.

    The final reward (shaped curve + wrong-direction term) is capped at
    +1.0 before the global ``weight`` is applied.
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

        # Clamp the °C error at floor_diff_celsius so the negative tail of
        # the reward stays bounded. Without this, large drift (only reachable
        # when early termination is disabled) produces -d² spikes that can
        # destabilise SAC gradients.
        diff_celsius = min(diff_celsius, self.floor_diff_celsius)

        # Smooth comfort curve
        if diff_celsius < self.zero_reward_diff_celsius:
            # Inside the positive-reward region, the curve is a concave parabola peaking at 1.0 when diff=0.
            reward = -self.k * (diff_celsius ** 2) + 1
        else:
            reward = -1.0 * np.abs(diff_celsius) + self.zero_reward_diff_celsius # Linear negative tail beyond zero-reward threshold

        # Wrong-direction penalty: heating when too hot, or cooling when too cold
        # HP action: negative = cooling, positive = heating
        # Scaled by |°C error| (clamped to floor_diff_celsius) so the
        # coefficient has a stable physical meaning across temp_abs_max values.
        if "a_hp" in actions:
            hp_action = float(np.atleast_1d(actions["a_hp"])[0])
            energy = abs(hp_action)
            temp_error_celsius = (actual_temp - desired_temp) * temp_abs_max  # +ve = too hot

            wrong = energy > 0 and (
                (temp_error_celsius > 0 and hp_action > 0) or   # too hot but heating
                (temp_error_celsius < 0 and hp_action < 0)      # too cold but cooling
            )
            if wrong:
                error_magnitude_celsius = min(abs(temp_error_celsius), self.floor_diff_celsius)
                reward += self.wrong_direction_penalty * energy * error_magnitude_celsius

        return self.weight * min(reward, 1.0), self.weight * self.max_reward_in_step


# Register TempReward with the component registry
ComponentRegistry.register('reward', TempReward)
