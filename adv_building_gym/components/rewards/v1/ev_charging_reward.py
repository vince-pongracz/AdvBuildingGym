"""EV charging SoC tracking reward function."""

import logging

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class EVChargingReward(RewardFunction):
    """Reward for keeping the EV SoC close to its target.

    While connected: full reward within ``diff_threshold`` of target, else exponential
    decay. Disconnected steps give 0 reward.

    Terminal events (voted in ``should_terminate``, paid in ``get_reward``):
    - Disconnect: compare reached SoC vs session target — within ``disconnect_soc_tolerance``
      → ``success_reward`` else ``failure_penalty``; episode ends either way.
    - Min-curve: while connected and the session is *active* (target reachable), SoC below
      ``s_evc_soc_min`` → ``min_curve_violation_penalty`` + end. 
      When the session is inactive (target was unreachable
      from connect time) the corridor is suppressed and the agent should
      simply charge as fast as it can.
    """

    def __init__(self,
                weight: float,
                disconnect_soc_tolerance: float,
                name: str = "ev_charging_reward",
                diff_threshold: float = 0.02,
                soc_diff_multiplier: float = 5.0,
                success_reward: float = 10.0,
                failure_penalty: float = -100.0,
                min_curve_violation_penalty: float = -100.0) -> None:
        """
        Args:
            weight: Reward weight for multi-objective optimisation.
            disconnect_soc_tolerance: SoC tolerance at disconnect time
                inside which the session is judged a success.  No default
                -- a sensible starting value is 0.05.
            name: Reward function identifier.
            diff_threshold: SoC difference below which full reward is given.
                With a 60 kWh battery the default 0.02 corresponds to ~1.2 kWh.
            soc_diff_multiplier: Exponential decay rate.  Higher values
                penalise deviations more sharply.
            success_reward: Terminal reward when the EV disconnects within
                ``disconnect_soc_tolerance`` of target.
            failure_penalty: Terminal penalty when the EV disconnects too
                far from target.
            min_curve_violation_penalty: Terminal penalty when the SoC
                falls below the lazy min curve while connected.
        """
        super().__init__(weight, name)
        self.diff_threshold = diff_threshold
        self.soc_diff_multiplier = soc_diff_multiplier
        self.disconnect_soc_tolerance = disconnect_soc_tolerance
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.min_curve_violation_penalty = min_curve_violation_penalty

    def _disconnect_verdict(self, state, next_state) -> tuple[bool, bool]:
        """Detect+judge an EV disconnect from the (s, s') transition.

        Inferred from the connected flag flipping s→s'. The charger zeroes the obs on
        disconnect, so the reached SoC and target are read from ``state``.
        Returns ``(just_disconnected, success)``.
        """
        was_connected = float(state["s_evc_connected"][0]) >= 0.5
        now_connected = float(next_state["s_evc_connected"][0]) >= 0.5
        if not (was_connected and not now_connected):
            return False, False
        achieved_soc = float(state["s_evc_soc"][0])
        target_soc = float(state["s_evc_target_soc"][0])
        success = abs(achieved_soc - target_soc) <= self.disconnect_soc_tolerance
        return True, success

    def should_terminate(self, actions, state, next_state, info: dict | None = None) -> bool:
        if info is None:
            return False
        # Disconnect judgement: a detach this step ends the episode on failure.
        just_disconnected, success = self._disconnect_verdict(state, next_state)
        if just_disconnected:
            if not success:
                logger.info(
                    "[%s] terminal step on EV disconnect (FAILURE): SoC %.3f vs target %.3f (tol %.3f, step %s)",
                    self.name, float(state["s_evc_soc"][0]), float(state["s_evc_target_soc"][0]),
                    self.disconnect_soc_tolerance, info.get("iteration", "?"),
                )
                return True
            return False

        # Min-curve violation while the session is feasible and still connected.
        if float(next_state["s_evc_session_target_feasible"][0]) > 0.5 and float(next_state["s_evc_connected"][0]) >= 0.5:
            current_soc = float(next_state["s_evc_soc"][0])
            soc_min = float(next_state["s_evc_soc_min"][0])
            if current_soc < soc_min:
                logger.info(
                    "[%s] terminal step: SoC %.3f below min-curve %.3f (step %s)",
                    self.name, current_soc, soc_min, info.get("iteration", "?"),
                )
                return True
        return False

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        allow_term = info.get("allow_early_termination", True) if info is not None else True

        # Disconnect first: judge SoC reached before detach vs target. Skipped when
        # early termination is off (terminal success/failure don't fit a soft stream);
        # the disconnect step then falls through to the not-connected branch.
        just_disconnected, success = self._disconnect_verdict(state, next_state)
        if allow_term and just_disconnected:
            if success:
                return self.weight * self.success_reward
            return self.weight * self.failure_penalty

        if float(next_state["s_evc_connected"][0]) < 0.5:  # not connected: no reward
            return 0.0

        current_soc = float(next_state["s_evc_soc"][0])
        target_soc = float(next_state["s_evc_target_soc"][0])

        # Min-curve: enforced only for a feasible session (reachable target). Reading the
        # charger's corridor key keeps this reward a pure judge. Skipped when early
        # termination is off → falls through to the SoC-band reward.
        if allow_term and float(next_state["s_evc_session_target_feasible"][0]) > 0.5:
            soc_min = float(next_state["s_evc_soc_min"][0])
            if current_soc < soc_min:
                return self.weight * self.min_curve_violation_penalty

        soc_diff = abs(current_soc - target_soc)
        if soc_diff < self.diff_threshold:
            reward = 1.0
        else:
            reward = float(np.exp(-soc_diff * self.soc_diff_multiplier))

        return self.weight * reward


ComponentRegistry.register('reward', EVChargingReward)
