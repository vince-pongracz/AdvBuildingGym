import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# Info-dict keys used to persist recovery state across steps without
# storing mutable state on the reward object itself.  Using the info
# dict (like TempReward) ensures counters reset naturally at episode
# boundaries when the environment clears _component_info.
_STEP_KEY = "operator_reward_step"
_LAST_VIOLATION_KEY = "operator_reward_last_violation_step"


class OperatorEnergyControlReward(RewardFunction):
    """Reward function for respecting grid operator energy consumption limits.

    Three-zone reward based on consumption ratio (grid_power / operator_limit):

    - Below 90% of limit: full reward (1.0)
    - 90%-100% of limit: exponential decay from 1.0 towards 0
      using exp(-5 * (ratio - 0.9) / 0.1), where the scale factor 5
      gives exp(-5) ~ 0.007 at the limit boundary
    - Above limit: harsh_penalty (default -4.0), followed by an
      exponential recovery period of ``recovery_steps`` steps where
      the reward follows harsh_penalty * exp(-rate * k), decaying
      from harsh_penalty towards 0.

    Reads ``net_power_kW`` from the ``info`` dict and the per-episode
    operator limit from ``ctxt_operator_max_power_kW`` in the state dict
    (published by the OperatorEnergyControl statesource), so the kW limit
    has a single source of truth in the env config.

    Termination is exposed via the ``should_terminate`` ABC hook (Phase-1
    pre-pass in the env). ``get_reward`` no longer mutates the info dict.
    """

    # Scale factor for the exponential decay in the transition zone.
    # exp(-5) ~ 0.007, so reward nearly reaches 0 right at the limit.
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
            soft_threshold_pct: Fraction of operator limit below which reward is 1.0
                (default 0.9 = 90%). Must be in (0, 1).
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
        # Recovery rate chosen so that at step=recovery_steps the ceiling
        # is ~1% of harsh_penalty: exp(-rate * N) ~ 0.01 => rate = ln(100)/N
        # Link: standard exponential decay, solving for 99% recovery
        self._recovery_rate = np.log(100.0) / max(recovery_steps, 1)

    @staticmethod
    def _compute_ratio(states, info) -> float | None:
        """Net-power-to-operator-limit ratio, or None when limit is unavailable.

        Centralised so should_terminate and get_reward never disagree.
        """
        ctxt = states.get("ctxt_operator_max_power_kW")
        if ctxt is None:
            return None
        operator_limit_kW = float(ctxt[0])
        if operator_limit_kW <= 0:
            return None
        net_power_kW = float(info.get("net_power_kW", 0.0))
        return net_power_kW / operator_limit_kW

    def should_terminate(self, actions, states, info: dict | None = None) -> bool:
        if info is None:
            return False
        ratio = self._compute_ratio(states, info)
        if ratio is None:
            return False
        if ratio > self.terminate_threshold_pct:
            logger.info(
                "[%s] terminal step on operator-limit breach: ratio %.3f > %.3f "
                "(net %.3f kW, limit %.3f kW, step %s)",
                self.name, ratio, self.terminate_threshold_pct,
                float(info.get("net_power_kW", 0.0)),
                float(states["ctxt_operator_max_power_kW"][0]),
                info.get("iteration", "?"),
            )
            return True
        return False

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        """Calculate reward based on grid power consumption vs operator limit."""
        if info is None:
            logger.warning("OperatorEnergyControlReward: info dict is None, returning 0")
            return 0.0, (self.weight * self.max_reward_in_step)

        # Read and advance per-episode step counter from info dict.
        # Resets to 0 at episode start because _component_info is cleared.
        step = info.get(_STEP_KEY, 0) + 1
        info[_STEP_KEY] = step
        last_violation_step = info.get(_LAST_VIOLATION_KEY, -self.recovery_steps)

        ratio = self._compute_ratio(states, info)

        # No usable operator limit -- skip this reward, don't know what to punish
        if ratio is None:
            logger.error("E usage ratio can't be computed")
            return 0.0, 0.0

        allow_term = info.get("allow_early_termination", False)
        if allow_term and ratio > self.terminate_threshold_pct:
            # should_terminate already voted to end the episode in Phase 1;
            # emit the configured terminal penalty here.  When early
            # termination is disabled, we fall through to the regular
            # over-limit branch (harsh_penalty + recovery) below.
            return float(self.weight * self.terminate_penalty), (self.weight * self.max_reward_in_step)

        if self.soft_threshold_pct < ratio <= 1.0:
            # Transition zone: exponential decay from 0.0 towards -1.0
            t = (ratio - self.soft_threshold_pct) / (1.0 - self.soft_threshold_pct)
            reward = float(np.exp(-self._DECAY_SCALE * t)) - 1.0
        else:
            # Between operator limit and terminate threshold: harsh penalty
            # and mark violation so the recovery zone applies on subsequent
            # steps.
            info[_LAST_VIOLATION_KEY] = step
            return float(self.weight * self.harsh_penalty), (self.weight * self.max_reward_in_step)

        # During recovery: override reward with an exponential curve from
        # harsh_penalty towards 0.  The agent earns a negative (but shrinking)
        # reward for recovery_steps steps, then normal rewarding resumes.
        steps_since_violation = step - last_violation_step
        if steps_since_violation <= self.recovery_steps:
            reward = float(self.harsh_penalty * np.exp(-self._recovery_rate * steps_since_violation))

        return float(self.weight * reward), (self.weight * self.max_reward_in_step)


# Register OperatorEnergyControlReward with the component registry
ComponentRegistry.register('reward', OperatorEnergyControlReward)
