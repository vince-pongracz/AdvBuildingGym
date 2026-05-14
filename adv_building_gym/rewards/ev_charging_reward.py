"""EV charging SoC tracking reward function."""

import logging

import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class EVChargingReward(RewardFunction):
    """Reward for keeping the EV state-of-charge close to its target.

    While the EV is connected, gives full reward when ``|ev_soc - target|``
    is within ``diff_threshold`` and exponentially decaying reward outside
    that band. Disconnected periods contribute 0 reward and 0 max-step
    so the reward rate is not inflated.

    Terminal events (vote via ``should_terminate`` in the env's Phase-1
    pass; penalty/reward emitted by ``get_reward`` in Phase 2):

    - **Disconnect judgement**: on the step the EV disconnects, compare
      SoC against the snapshotted session target.  Within
      ``disconnect_soc_tolerance`` → ``success_reward``, else
      ``failure_penalty``; episode ends in either case.
    - **Min-curve violation**: while connected and the session is *active*
      (target reachable from start at max charge rate), if the SoC falls
      below ``s_ev_soc_min`` (the lazy back-from-target line published by
      the charger), apply ``min_curve_violation_penalty`` and end the
      episode.  When the session is inactive (target was unreachable
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

    def should_terminate(self, actions, states, info: dict | None = None) -> bool:
        if info is None:
            return False
        # Disconnect judgement: any EV detach this step ends the episode.
        if info.get("ev_just_disconnected", False):
            current_soc = float(states["s_ev_soc"][0])
            target_soc = float(info.get("ev_session_target_soc", 0.0))
            success = abs(current_soc - target_soc) <= self.disconnect_soc_tolerance
            if not success:
                logger.info(
                    "[%s] terminal step on EV disconnect (FAILURE): SoC %.3f vs target %.3f (tol %.3f, step %s)",
                    self.name, current_soc, target_soc, self.disconnect_soc_tolerance,
                    info.get("iteration", "?"),
                )
                return True
            else:
                return False

        # Min-curve violation while the session is active.
        if info.get("ev_session_active", False) and float(states["s_ev_connected"][0]) >= 0.5:
            current_soc = float(states["s_ev_soc"][0])
            soc_min = float(states["s_ev_soc_min"][0])
            if current_soc < soc_min:
                logger.info(
                    "[%s] terminal step: SoC %.3f below min-curve %.3f (step %s)",
                    self.name, current_soc, soc_min, info.get("iteration", "?"),
                )
                return True
        return False

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        max_step = self.weight * self.max_reward_in_step

        allow_term = info.get("allow_early_termination", True) if info is not None else True

        # Disconnect judgement runs first so the terminal verdict fires on
        # the same step the EV detaches, regardless of s_ev_connected which
        # has already flipped to 0.  Skipped entirely when the env disables
        # early termination — both success_reward and failure_penalty are
        # exceptional values that don't belong in a soft, non-terminating
        # reward stream.  In that mode the disconnect step falls through to
        # the ev_connected < 0.5 fallthrough below (no signal).
        if allow_term and info is not None and info.get("ev_just_disconnected", False):
            current_soc = float(states["s_ev_soc"][0])
            target_soc = float(info.get("ev_session_target_soc", 0.0))
            success = abs(current_soc - target_soc) <= self.disconnect_soc_tolerance
            if success:
                return self.weight * self.success_reward, max_step
            return self.weight * self.failure_penalty, max_step

        ev_connected = float(states["s_ev_connected"][0])
        if ev_connected < 0.5:
            return 0.0, 0.0

        current_soc = float(states["s_ev_soc"][0])
        target_soc = float(states["s_ev_target_soc"][0])

        # Min-curve violation: only enforced when the session is active
        # (reachable target).  The corridor obs key is published by the
        # charger; reading it here keeps the reward purely a judge.  When
        # early termination is disabled, the huge penalty is skipped and
        # the agent receives the regular SoC-band reward below.
        if allow_term and info is not None and info.get("ev_session_active", False):
            soc_min = float(states["s_ev_soc_min"][0])
            if current_soc < soc_min:
                return self.weight * self.min_curve_violation_penalty, max_step

        soc_diff = abs(current_soc - target_soc)
        if soc_diff < self.diff_threshold:
            reward = 1.0
        else:
            reward = float(np.exp(-soc_diff * self.soc_diff_multiplier))

        return self.weight * reward, max_step


ComponentRegistry.register('reward', EVChargingReward)
