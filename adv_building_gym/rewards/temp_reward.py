import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry


# Info-dict key used to persist the consecutive-violation counter across
# steps without storing state on the reward object itself.
_SUSTAINED_KEY = "temp_reward_consecutive_violations"


class TempReward(RewardFunction):
    """
    Temperature-based reward function.
    This reward encourages the agent to maintain the indoor temperature
    close to the desired user temperature setpoint.

    Supports five strictness mechanisms (all optional, all on by default):

    1. **Dead-zone removal**: ``diff_threshold=0`` eliminates the flat
       "good-enough" band so every deviation is penalised.
    2. **Steep exponential decay**: ``temp_const_multiplier`` controls how
       fast reward drops with temperature error.
    3. **Quadratic penalty blend**: ``quadratic_weight`` blends a quadratic
       term ``1 - (diff / max_diff)^2`` with the exponential, giving a
       sharper gradient near the setpoint.
    4. **Sustained-deviation penalty**: After ``sustained_steps_threshold``
       consecutive steps above ``diff_threshold``, an escalating penalty
       is subtracted (capped at ``sustained_penalty_cap``).  The counter
       is stored in the shared ``info`` dict; episode boundaries are
       detected via ``sim_hour`` so the reward remains stateless.
    5. **Proportional wrong-direction penalty**: Instead of a flat penalty,
       scales by ``|energy| * |temp_error|``, so larger mistakes are
       penalised more.
    """

    def __init__(
        self,
        weight: float,
        diff_threshold: float = 0.0,
        name: str = "temp_reward",
        wrong_direction_penalty: float = -1.0,
        proportional_wrong_direction: bool = True,
        temp_const_multiplier: float = 10.0,
        quadratic_weight: float = 0.5,
        max_norm_diff: float = 1.0,
        sustained_steps_threshold: int = 6,
        sustained_penalty_per_step: float = 0.05,
        sustained_penalty_cap: float = 0.5,
    ) -> None:
        """
        Args:
            weight: Global weight applied to the final reward.
            diff_threshold: Normalised temperature difference below which
                full reward is given.  Set to 0 to penalise every
                deviation (strict mode).
            wrong_direction_penalty: Base penalty when the HP mode works
                against the temperature error direction.
            proportional_wrong_direction: If True, scale the wrong-direction
                penalty by ``|energy| * |temp_error|`` instead of applying
                a flat value.
            temp_const_multiplier: Exponential decay rate.  Higher values
                penalise deviations more sharply.
            quadratic_weight: Blend factor in [0, 1].  0 = pure exponential,
                1 = pure quadratic.  The final comfort reward is
                ``(1 - qw) * exp_reward + qw * quad_reward``.
            max_norm_diff: Maximum expected normalised temperature
                difference, used to scale the quadratic term to [0, 1].
            sustained_steps_threshold: Number of consecutive steps the
                temperature error must exceed ``diff_threshold`` before
                the sustained-deviation penalty activates.
            sustained_penalty_per_step: Penalty increment for each step
                beyond the threshold.
            sustained_penalty_cap: Maximum sustained-deviation penalty
                (keeps the total reward bounded).
        """
        super().__init__(weight, name)
        self.diff_threshold = diff_threshold
        self.temp_const_multiplier = temp_const_multiplier
        self.wrong_direction_penalty = wrong_direction_penalty
        self.proportional_wrong_direction = proportional_wrong_direction
        self.quadratic_weight = np.clip(quadratic_weight, 0.0, 1.0)
        self.max_norm_diff = max_norm_diff
        self.sustained_steps_threshold = sustained_steps_threshold
        self.sustained_penalty_per_step = sustained_penalty_per_step
        self.sustained_penalty_cap = sustained_penalty_cap

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        actual_temp = float(states["temp_in_norm"][0])
        desired_temp = float(states["desired_temp_in_norm"][0])

        temp_diff = abs(actual_temp - desired_temp)

        # --- 1 & 2: comfort reward (exponential + optional dead-zone) ---
        if temp_diff < self.diff_threshold:
            exp_reward = 1.0
        else:
            exp_reward = float(np.exp(-temp_diff * self.temp_const_multiplier))

        # --- 3: quadratic penalty blend ---
        quad_reward = 1.0 - (min(temp_diff, self.max_norm_diff) / self.max_norm_diff) ** 2
        qw = self.quadratic_weight
        reward = (1.0 - qw) * exp_reward + qw * quad_reward

        # --- 4: sustained-deviation penalty ---
        # The consecutive-violation counter lives in the shared info dict
        # so the reward function itself remains stateless.  Episode
        # boundaries are detected via sim_hour: the first step() of an
        # episode sets sim_hour to exactly one control-step worth of hours,
        # so any value at or below that threshold resets the counter.
        if info is not None:
            sim_hour = float(states["sim_hour"][0])

            if sim_hour > 0.0:
                consecutive = info.get(_SUSTAINED_KEY, 0)
            else:
                consecutive = 0

            if temp_diff >= self.diff_threshold:
                consecutive += 1
            else:
                consecutive = 0

            info[_SUSTAINED_KEY] = consecutive

            if consecutive > self.sustained_steps_threshold:
                overshoot = consecutive - self.sustained_steps_threshold
                sustained_penalty = min(
                    overshoot * self.sustained_penalty_per_step,
                    self.sustained_penalty_cap,
                )
                reward -= sustained_penalty

        # --- 5: proportional wrong-direction penalty ---
        # HP mode convention: <0.4 = cooling, >0.6 = heating, [0.4, 0.6] = off
        if "HP_action" in actions:
            energy = float(np.atleast_1d(actions["HP_action"])[0])
            mode = float(np.atleast_1d(actions["HP_action"])[1])
            temp_error = actual_temp - desired_temp  # positive = too hot

            wrong = False
            if energy > 0:
                if temp_error > 0 and mode > 0.6:
                    wrong = True
                elif temp_error < 0 and mode < 0.4:
                    wrong = True

            if wrong:
                if self.proportional_wrong_direction:
                    # Scale by how much energy and how far off-target
                    penalty = self.wrong_direction_penalty * abs(energy) * abs(temp_error)
                else:
                    penalty = self.wrong_direction_penalty
                reward = penalty

        reward = float(np.clip(reward, -1.0, 1.0))
        return self.weight * reward, self.weight * self.max_reward


# Register TempReward with the component registry
ComponentRegistry.register('reward', TempReward)
