"""EV regulator reward: flat penalty whenever the EV charger overrides the requested action."""

import logging

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class EVRegulatorReward(RewardFunction):
    """Penalise EV charger actions the hardware has to override.

    ``LinearEVCharger.exec_action`` regulates the request away in three places, all of
    which this reward prices with the same flat ``penalty``:

    - **no EV plugged in** — the action is forced to 0 and the step returns early;
    - **V2G not permitted** (either V2G flag off, or SoC below ``target - v2g_playroom``)
      — a negative action is lifted to 0;
    - **SoC overshoot** — the tentative SoC leaves [0, 1] and the action is
      back-calculated to the value that just reaches the bound.

    The three are mutually exclusive per step (the disconnected branch returns before the
    others run), so at most one ``penalty`` is paid. ``next_state["s_evc_connected"]``
    separates the disconnected case from the connected ones for the diagnostics — it is
    published in the same step, after ``_check_schedule`` has applied the connect/disconnect.

    The requested (pre-override) action comes from ``info["requested_action"]``,
    snapshotted by ``AdvBuildingGym._execute_actions`` before the infras mutate the
    action dict.
    """

    def __init__(self,
                weight: float,
                name: str = "ev_regulator_reward",
                penalty: float = -1.0,
                tolerance: float = 1e-3,
                action_key: str = "a_lin_ev_charger") -> None:
        """
        Args:
            weight: Reward weight for multi-objective optimisation.
            penalty: Flat per-step reward (<= 0) paid when the action was overridden.
            tolerance: Absolute |requested - realised| below which the mismatch counts
                as float noise. The SoC-clipped branch back-calculates the action through
                energy -> SoC -> energy in float32, so an exact comparison would
                false-positive on unclipped steps.
            action_key: EV charger action key to watch.
        """
        super().__init__(weight, name)
        if penalty > 0:
            raise ValueError("penalty must be <= 0.")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0.")
        self.penalty = penalty
        self.tolerance = tolerance
        self.action_key = action_key

    def get_reward(self, actions, state, next_state, info: dict) -> float:
        requested_actions = info.get("requested_action") or {}
        if self.action_key not in requested_actions or self.action_key not in actions:
            return 0.0

        requested = float(np.atleast_1d(requested_actions[self.action_key])[0])
        realised = float(np.atleast_1d(actions[self.action_key])[0])
        overridden = abs(requested - realised) > self.tolerance
        # s' connection flag: the charger applies the schedule inside exec_action and
        # republishes s_evc_connected in the same step, so this is the state the override
        # was decided under.
        connected = float(next_state["s_evc_connected"][0]) >= 0.5

        self._publish_diagnostics(info, overridden=overridden, connected=connected)
        if not overridden:
            return 0.0

        logger.debug("[%s] EV action overridden: requested %.4f -> realised %.4f (connected=%s)",
                    self.name, requested, realised, connected)
        return self.weight * self.penalty

    def _publish_diagnostics(self, info: dict, *, overridden: bool, connected: bool) -> None:
        """Record per-step 0/1 flags into ``info["reward_diagnostics"]`` (cleared each pass).

        Split so the two failure modes are separable in TensorBoard: acting on an absent
        EV is a different policy error than a connected EV regulating the request down
        (SoC bound hit, or V2G not permitted). Summed per episode by EpisodeMetricsCallback.
        """
        diag = info.setdefault("reward_diagnostics", {})
        diag[f"{self.name}/action_on_disconnected_ev"] = float(overridden and not connected)
        diag[f"{self.name}/action_overridden_on_connected_ev"] = float(overridden and connected)


ComponentRegistry.register('reward', EVRegulatorReward)
