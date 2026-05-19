"""EV charging SoC tracking reward function (V0)."""

import logging

import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class EVChargingRewardV0(RewardFunction):
    """EV charging reward (V0): dense connected-step shaping + sparse session verdicts.

    Connected, normal step (dense, ``[0, 1]``):
        - ``|SoC - target| < diff_threshold`` → ``1.0``
        - else → ``exp(-soc_diff_multiplier * |SoC - target|)``

    Sparse one-shot events fire with magnitude scaled to the
    **current charging session length** (number of connected steps
    accumulated since the last connect), so the terminal verdict is
    commensurate with the cumulative dense return for that session:

    - **Disconnect step** (``info["ev_just_disconnected"]``):
      success (``|SoC - target| <= disconnect_soc_tolerance``) →
      ``+session_steps``; else → ``-session_steps``.
    - **Min-curve violation** while ``info["ev_session_active"]`` and
      ``s_ev_soc < s_ev_soc_min``: emits ``-session_steps``.

    Explicit constructor floats for ``success_reward``,
    ``failure_penalty``, ``min_curve_violation_penalty`` override the
    session-length default. ``None`` ⇒ use session-length magnitude.

    No ``should_terminate`` logic.

    Disconnected (and not the just-disconnected step): ``(0.0, 0.0)``.
    """

    _exclude_params = {"_session_steps"}

    def __init__(self,
                weight: float,
                disconnect_soc_tolerance: float,
                name: str = "ev_charging_reward_v0",
                diff_threshold: float = 0.02,
                soc_diff_multiplier: float = 5.0,
                success_reward: float | None = None,
                failure_penalty: float | None = None,
                min_curve_violation_penalty: float | None = None) -> None:
        super().__init__(weight, name)
        self.diff_threshold = diff_threshold
        self.soc_diff_multiplier = soc_diff_multiplier
        self.disconnect_soc_tolerance = disconnect_soc_tolerance
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.min_curve_violation_penalty = min_curve_violation_penalty
        self._session_steps: int = 0

    def on_reset(self, states, info: dict | None = None) -> None:
        self._session_steps = 0

    def _resolve_signed(self, override: float | None, sign: int, magnitude: int) -> float:
        # sign: +1 for success, -1 for failure / violation.
        # When override is provided, use it as-is (the caller already encodes
        # sign in the override value, matching the V1 class's constructor
        # convention).
        if override is not None:
            return float(override)
        return float(sign * max(magnitude, 1))

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        if info is not None and info.get("ev_just_disconnected", False):
            current_soc = float(states["s_ev_soc"][0])
            target_soc = float(info.get("ev_session_target_soc", 0.0))
            success = abs(current_soc - target_soc) <= self.disconnect_soc_tolerance
            magnitude = max(self._session_steps, 1)
            max_step = self.weight * float(magnitude)

            if success:
                value = self._resolve_signed(self.success_reward, +1, magnitude)
            else:
                value = self._resolve_signed(self.failure_penalty, -1, magnitude)

            self._session_steps = 0
            return self.weight * value, max_step

        ev_connected = float(states["s_ev_connected"][0])
        if ev_connected < 0.5:
            return 0.0, 0.0

        current_soc = float(states["s_ev_soc"][0])
        target_soc = float(states["s_ev_target_soc"][0])

        if info is not None and info.get("ev_session_active", False):
            soc_min = float(states["s_ev_soc_min"][0])
            if current_soc < soc_min:
                magnitude = max(self._session_steps, 1)
                max_step = self.weight * float(magnitude)
                value = self._resolve_signed(self.min_curve_violation_penalty, -1, magnitude)
                return self.weight * value, max_step

        # Regular connected step: dense [0, 1] shaping.
        self._session_steps += 1
        soc_diff = abs(current_soc - target_soc)
        if soc_diff < self.diff_threshold:
            reward = 1.0
        else:
            reward = float(np.exp(-soc_diff * self.soc_diff_multiplier))

        return self.weight * reward, self.weight * self.max_reward_in_step


ComponentRegistry.register('reward', EVChargingRewardV0)
