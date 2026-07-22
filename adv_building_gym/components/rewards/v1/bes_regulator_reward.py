"""BES regulator reward: flat penalty whenever the battery overrides the requested action."""

import logging

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class BESRegulatorReward(RewardFunction):
    """Penalise battery actions the hardware has to clip (SoC overshoot).

    ``BatteryLinear.exec_action`` clips the tentative SoC to ``[soc_min, soc_max]`` and
    rewrites ``actions["a_battery"]`` to the fraction actually realised, so the request
    is silently regulated away — charging a full pack or discharging an empty one costs
    the agent nothing and stays invisible in the action trace. This reward prices that
    gap: whenever |requested - realised| exceeds ``tolerance``, pay ``penalty`` (flat,
    independent of how far the request overshot).

    The requested (pre-clip) action comes from ``info["requested_action"]``, snapshotted
    by ``AdvBuildingGym._execute_actions`` before the infras mutate the action dict.
    """

    def __init__(self,
                weight: float,
                name: str = "bes_regulator_reward",
                penalty: float = -1.0,
                tolerance: float = 1e-3,
                action_key: str = "a_battery") -> None:
        """
        Args:
            weight: Reward weight for multi-objective optimisation.
            penalty: Flat per-step reward (<= 0) paid when the action was clipped.
            tolerance: Absolute |requested - realised| below which the mismatch counts
                as float noise. The realised action is a round trip through
                energy -> SoC -> energy in float32, so an exact comparison would
                false-positive on unclipped steps.
            action_key: Battery action key to watch.
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
        overshot = abs(requested - realised) > self.tolerance

        self._publish_diagnostics(info, overshot=overshot)
        if not overshot:
            return 0.0

        logger.debug("[%s] battery action clipped: requested %.4f -> realised %.4f (SoC %.3f)",
                    self.name, requested, realised, float(next_state["s_battery_soc"][0]))
        return self.weight * self.penalty

    def _publish_diagnostics(self, info: dict, *, overshot: bool) -> None:
        """Record the per-step 0/1 clip flag into ``info["reward_diagnostics"]``
        (cleared each pass); summed per episode by EpisodeMetricsCallback."""
        diag = info.setdefault("reward_diagnostics", {})
        diag[f"{self.name}/action_overshot"] = float(overshot)


ComponentRegistry.register('reward', BESRegulatorReward)
