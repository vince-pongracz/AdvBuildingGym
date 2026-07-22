"""EV charging session reward: sparse disconnect verdict + bounded charge-progress shaping."""

import logging

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class EVChargingSessionReward(RewardFunction):
    """Session-verdict EV reward designed to coexist with ``EconomicReward``.

    Dense (connected steps): ``charge_progress_scale`` x the step's first-time SoC
    progress toward target, normalised by the charge the session needs — over a whole
    session the shaping sums to at most ``charge_progress_scale`` (default 0.5), far
    below the economic reward's per-episode scale, so WHEN to charge stays priced by
    ``EconomicReward`` while this term only guides exploration toward charging at all.
    A progress watermark blocks charge/discharge (V2G) cycling from re-earning credit;
    discharging, idling and SoC above target earn 0.

    Sparse (disconnect step): reached SoC >= target - ``disconnect_soc_tolerance``
    -> ``success_reward``, else ``failure_penalty``. The check is one-sided on purpose:
    overshoot is not a session failure, its cost is the energy bill priced by
    ``EconomicReward``. Both magnitudes are fixed (not scaled by session length) so the
    critic never sees the hundreds-sized spikes ``EVChargingRewardV0`` produced.

    No ``should_terminate``: pure reward stream; an episode ending while the EV is
    still connected yields no verdict.
    """

    _exclude_params = {"_watermark", "_session_soc_needed"}

    def __init__(self,
                weight: float,
                name: str = "ev_charging_session_reward",
                success_reward: float = 30.0,
                failure_penalty: float = -60.0,
                disconnect_soc_tolerance: float = 0.05,
                charge_progress_scale: float = 0.5) -> None:
        """
        Args:
            weight: Reward weight for multi-objective optimisation.
            success_reward: One-shot reward when the EV disconnects at/above
                target - tolerance. Sized to dominate a whole episode's economic
                return so the session target stays the primary objective.
            failure_penalty: One-shot penalty (<= 0) when the EV disconnects below
                target - tolerance. Must exceed (in magnitude) the largest economic
                gain skippable charging could yield, so failing is never profitable.
            disconnect_soc_tolerance: SoC shortfall below target still judged a success.
            charge_progress_scale: Total dense shaping paid out for fully covering the
                session's needed charge (start SoC -> target).
        """
        super().__init__(weight, name)
        if success_reward < 0:
            raise ValueError("success_reward must be >= 0.")
        if failure_penalty > 0:
            raise ValueError("failure_penalty must be <= 0.")
        if not 0.0 <= disconnect_soc_tolerance < 1.0:
            raise ValueError("disconnect_soc_tolerance must be in [0, 1).")
        if charge_progress_scale < 0:
            raise ValueError("charge_progress_scale must be >= 0.")
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.disconnect_soc_tolerance = disconnect_soc_tolerance
        self.charge_progress_scale = charge_progress_scale
        self._watermark: float | None = None
        self._session_soc_needed: float = 0.0

    def on_reset(self, states, info: dict) -> None:
        self._watermark = None
        self._session_soc_needed = 0.0

    def get_reward(self, actions, state, next_state, info: dict) -> float:
        was_connected = float(state["s_evc_connected"][0]) >= 0.5
        now_connected = float(next_state["s_evc_connected"][0]) >= 0.5

        # Disconnect verdict: the charger zeroes the s_evc_* obs on detach, so the
        # reached SoC and target are judged from ``state`` (pre-detach values).
        if was_connected and not now_connected:
            achieved_soc = float(state["s_evc_soc"][0])
            target_soc = float(state["s_evc_target_soc"][0])
            self._watermark = None
            if achieved_soc >= target_soc - self.disconnect_soc_tolerance:
                return self.weight * self.success_reward
            logger.debug("[%s] session FAILURE: SoC %.3f < target %.3f - tol %.3f",
                        self.name, achieved_soc, target_soc, self.disconnect_soc_tolerance)
            return self.weight * self.failure_penalty

        if not now_connected:
            return 0.0

        current_soc = float(next_state["s_evc_soc"][0])
        target_soc = float(next_state["s_evc_target_soc"][0])

        if self._watermark is None:
            # First connected observation anchors the session. s' already includes the
            # connect step's charging (LinearEVCharger.exec_action updates SoC before
            # update_state publishes it), so this step itself earns 0.
            self._watermark = min(current_soc, target_soc)
            self._session_soc_needed = max(target_soc - self._watermark, 0.0)
            return 0.0

        if self._session_soc_needed <= 0.0:
            return 0.0

        progress = min(current_soc, target_soc) - self._watermark
        if progress <= 0.0:
            return 0.0
        self._watermark = min(current_soc, target_soc)
        return self.weight * self.charge_progress_scale * progress / self._session_soc_needed


ComponentRegistry.register('reward', EVChargingSessionReward)
