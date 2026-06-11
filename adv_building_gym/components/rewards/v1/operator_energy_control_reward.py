import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Info-dict keys persisting recovery state across steps (not on the reward object),
# so counters reset at episode boundaries when the env clears _component_info.
_STEP_KEY = "operator_reward_step"
_LAST_VIOLATION_KEY = "operator_reward_last_violation_step"


class OperatorEnergyControlReward(RewardFunction):
    """Symmetric grid-operator power-limit penalty (draw and feed-in alike).

    ``ratio = |net_power_kW| / ctxt_operator_max_power_kW`` (EnergyTracker sign: net>0 export, <0 consume):
    - safe (ratio ≤ soft_threshold_pct): 0.
    - warning (soft_threshold_pct < ratio ≤ 1): ``exp(-_DECAY_SCALE*(ratio-soft)/(1-soft)) - 1``, 0 → ~-1 at the limit.
    - over-limit (1 < ratio ≤ terminate_threshold_pct): flat ``harsh_penalty`` (-4); marks a violation, after
      which the next ``recovery_steps`` warning rewards are overridden by ``harsh_penalty*exp(-rate*k)``
      (rate = ln(100)/recovery_steps → ~1% of harsh_penalty at k=recovery_steps).
    - breach (ratio > terminate_threshold_pct, default 1.1): ``should_terminate`` ends the episode; the
      terminal step pays ``terminate_penalty`` (-100) when ``allow_early_termination`` else the over-limit branch.

    Penalty-only. Limit from ``ctxt_operator_max_power_kW``, ``net_power_kW``
    from info. Per-episode counters live in ``info`` (reset when the env clears _component_info); get_reward
    writes them but mutates no termination flags.
    """

    # Scale factor for the exponential decay in the transition zone.
    # exp(-5) ~ 0.007, so reward nearly reaches -1 right at the limit.
    _DECAY_SCALE: float = 5.0

    def __init__(self,
                weight: float,
                name: str = "operator_energy_control_reward",
                harsh_penalty: float = -4.0,
                soft_threshold_pct: float = 0.9,
                recovery_steps: int = 3,
                terminate_threshold_pct: float = 1.1,
                terminate_penalty: float = -100.0,
                ) -> None:
        """Initialize OperatorEnergyControlReward.

        Args:
            weight: Reward weight (scaling factor).
            name: Reward function name.
            harsh_penalty: Flat penalty when consumption exceeds the operator limit.
            soft_threshold_pct: Fraction of operator limit below which the reward
                is 0 (default 0.9 = 90%). Above this and up to the limit the
                reward decays smoothly from 0 to ~-1. Must be in (0, 1).
            recovery_steps: Number of steps after a harsh penalty during which
                the reward is suppressed and exponentially recovers towards 0
                before returning to normal (default 3).
            terminate_threshold_pct: Ratio above which the episode is terminated
                (default 1.1). Must be > 1.0.
            terminate_penalty: Reward emitted on the terminating step.
        """
        super().__init__(weight, name)
        self.harsh_penalty = harsh_penalty
        if not 0.0 < soft_threshold_pct < 1.0:
            raise ValueError("soft_threshold_pct must be in (0, 1)")
        self.soft_threshold_pct = soft_threshold_pct
        self.recovery_steps = recovery_steps
        if terminate_threshold_pct <= 1.0:
            raise ValueError("terminate_threshold_pct must be > 1.0")
        self.terminate_threshold_pct = terminate_threshold_pct
        self.terminate_penalty = terminate_penalty
        # rate so exp(-rate*N) ~ 0.01 at N=recovery_steps → rate = ln(100)/N (99% recovery)
        self._recovery_rate = np.log(100.0) / max(recovery_steps, 1)

    @staticmethod
    def _compute_ratio(states, info) -> float | None:
        """|net|/operator-limit ratio, or None if no limit. Symmetric (uses |net_power_kW|);
        centralised so should_terminate and get_reward agree."""
        ctxt = states.get("ctxt_operator_max_power_kW")
        if ctxt is None:
            return None
        operator_limit_kW = float(ctxt[0])
        if operator_limit_kW <= 0:
            return None
        net_power_kW = float(info.get("net_power_kW", 0.0))
        return abs(net_power_kW) / operator_limit_kW

    def should_terminate(self, actions, state, next_state, info: dict | None = None) -> bool:
        if info is None:
            return False
        # operator limit from state (static ctxt); net_power_kW from info
        ratio = self._compute_ratio(state, info)
        if ratio is None:
            return False
        if ratio > self.terminate_threshold_pct:
            logger.info(
                "[%s] terminal step on operator-limit breach: ratio %.3f > %.3f "
                "(net %.3f kW, limit %.3f kW, step %s)",
                self.name, ratio, self.terminate_threshold_pct,
                float(info.get("net_power_kW", 0.0)),
                float(state["ctxt_operator_max_power_kW"][0]),
                info.get("iteration", "?"),
            )
            return True
        return False

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        """Calculate reward based on grid power consumption vs operator limit."""
        if info is None:
            logger.warning("OperatorEnergyControlReward: info dict is None, returning 0")
            return 0.0

        ratio = self._compute_ratio(state, info)
        if ratio is None:
            logger.error("E usage ratio can't be computed")
            return 0.0

        # per-episode step counter (resets when env clears _component_info)
        step = info[_STEP_KEY] = info.get(_STEP_KEY, 0) + 1

        # terminal breach — paid only with early termination; else falls through
        if ratio > self.terminate_threshold_pct and info.get("allow_early_termination", False):
            return float(self.weight * self.terminate_penalty)

        # over-limit (also breach with early-term off): flat penalty + mark violation
        if ratio > 1.0:
            info[_LAST_VIOLATION_KEY] = step
            return float(self.weight * self.harsh_penalty)

        # Safe zone.
        if ratio <= self.soft_threshold_pct:
            return 0.0

        # warning zone: smooth decay 0 → ~-1, overridden by the recovery curve
        # while within recovery_steps of the last violation
        steps_since_violation = step - info.get(_LAST_VIOLATION_KEY, -self.recovery_steps)
        if steps_since_violation <= self.recovery_steps:
            reward = self.harsh_penalty * np.exp(-self._recovery_rate * steps_since_violation)
        else:
            t = (ratio - self.soft_threshold_pct) / (1.0 - self.soft_threshold_pct)
            reward = np.exp(-self._DECAY_SCALE * t) - 1.0

        return float(self.weight * reward)


# register with ComponentRegistry
ComponentRegistry.register('reward', OperatorEnergyControlReward)
