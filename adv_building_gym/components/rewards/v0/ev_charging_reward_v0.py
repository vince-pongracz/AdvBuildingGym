"""EV charging SoC tracking reward function (V0)."""

import logging

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class EVChargingRewardV0(RewardFunction):
    """EV charging reward (V0): dense connected-step shaping + sparse session verdicts.

    Connected step (dense [0, 1]): 1.0 within ``diff_threshold`` of target, else
    ``exp(-soc_diff_multiplier * |SoC - target|)``.

    Sparse one-shots scaled to the session length (connected steps since last connect):
    - Disconnect: success (within ``disconnect_soc_tolerance``) → ``+session_steps``, 
    else ``-session_steps``.
    - Min-curve violation (``s_evc_session_target_feasible`` > 0.5 and ``s_evc_soc < s_evc_soc_min``)
    → ``-session_steps``.

    Explicit ``success_reward`` / ``failure_penalty`` / ``min_curve_violation_penalty`` override the
    session-length default (``None`` = use it). No ``should_terminate``; disconnected step → (0, 0).
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
        # sign: +1 success, -1 failure/violation. override (if given) is used as-is
        # (caller encodes the sign, matching the V1 ctor convention).
        if override is not None:
            return float(override)
        return float(sign * max(magnitude, 1))

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        # Disconnect this step (connected s → not s'). Charger zeroes s_evc_soc on detach,
        # so judge the SoC reached before leaving (both from ``state``) vs target.
        was_connected = float(state["s_evc_connected"][0]) >= 0.5
        now_connected = float(next_state["s_evc_connected"][0]) >= 0.5
        # disconnect step
        if was_connected and not now_connected:
            achieved_soc = float(state["s_evc_soc"][0])
            target_soc = float(state["s_evc_target_soc"][0])
            success = abs(achieved_soc - target_soc) <= self.disconnect_soc_tolerance
            magnitude = max(self._session_steps, 1)

            if success:
                value = self._resolve_signed(self.success_reward, +1, magnitude)
            else:
                value = self._resolve_signed(self.failure_penalty, -1, magnitude)

            self._session_steps = 0
            return self.weight * value

        if not now_connected:
            return 0.0

        current_soc = float(next_state["s_evc_soc"][0])
        target_soc = float(next_state["s_evc_target_soc"][0])

        if float(next_state["s_evc_session_target_feasible"][0]) > 0.5:
            soc_min = float(next_state["s_evc_soc_min"][0])
            if current_soc < soc_min:
                magnitude = max(self._session_steps, 1)
                value = self._resolve_signed(self.min_curve_violation_penalty, -1, magnitude)
                return self.weight * value

        # Regular connected step: dense [0, 1] shaping.
        self._session_steps += 1
        soc_diff = abs(current_soc - target_soc)
        if soc_diff < self.diff_threshold:
            reward = 1.0
        else:
            reward = float(np.exp(-soc_diff * self.soc_diff_multiplier))

        return self.weight * reward


ComponentRegistry.register('reward', EVChargingRewardV0)
