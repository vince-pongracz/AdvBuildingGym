import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Info-dict keys used to persist recovery state across steps without
# storing mutable state on the reward object itself.  Using the info
# dict (like TempReward) ensures counters reset naturally at episode
# boundaries when the environment clears _component_info.
_STEP_KEY = "operator_reward_step"
_LAST_VIOLATION_KEY = "operator_reward_last_violation_step"


class OperatorEnergyControlReward(RewardFunction):
    """Reward function for respecting a symmetric grid-operator power limit.

    The grid connection limit applies in BOTH directions: excessive draw
    AND excessive feed-in are punished equally. The decision variable is
    ``ratio = |net_power_kW| / ctxt_operator_max_power_kW`` (canonical
    sign convention from ``EnergyTracker``: ``net > 0`` = export, ``< 0``
    = consumption).

    - ``soft_threshold_pct < ratio <= 1.0`` (warning zone):
      ``exp(-_DECAY_SCALE * (ratio - soft_threshold_pct) / (1 - soft_threshold_pct)) - 1.0``,
      decaying from 0 towards -1 as |net| approaches the limit (with the
      default scale of 5, ``exp(-5) ~ 0.007`` so the floor is essentially
      -1 at the limit boundary).
    - ``1.0 < ratio <= terminate_threshold_pct`` (over-limit):
      ``harsh_penalty`` (default -4.0). The step is marked as a
      violation; on the next ``recovery_steps`` steps the warning-zone
      reward is overridden by an exponential recovery curve
      ``harsh_penalty * exp(-rate * k)`` decaying from ``harsh_penalty``
      towards ~0 (with ``rate = ln(100) / recovery_steps`` so the
      ceiling reaches ~1% of ``harsh_penalty`` at step ``k = recovery_steps``).
    - ``ratio > terminate_threshold_pct`` (default 1.1): the episode
      is ended via the ``should_terminate`` hook (Phase-1 pre-pass in
      the env). When the env publishes ``allow_early_termination=True``
      in ``info``, the terminal step pays ``terminate_penalty``
      (default -100); otherwise the over-limit branch above applies.
    - ``ratio <= soft_threshold_pct`` (safe zone): reward 0.

    This function never emits a positive reward, so ``max_reward_in_step``
    is 0 -- it acts as a pure penalty in the multi-objective sum.

    The kW limit lives on the ``OperatorEnergyControl`` statesource and is
    read here via ``ctxt_operator_max_power_kW``; ``net_power_kW`` comes
    from the env's ``_component_info`` dict. Per-episode counters
    (``operator_reward_step``, ``operator_reward_last_violation_step``)
    are stored in ``info`` so they reset automatically at episode
    boundaries when the env clears ``_component_info``. ``get_reward``
    writes those two counters but does not mutate any termination flags.
    """

    # Scale factor for the exponential decay in the transition zone.
    # exp(-5) ~ 0.007, so reward nearly reaches -1 right at the limit.
    _DECAY_SCALE: float = 5.0

    # Penalty-only reward: best achievable per step is 0.
    max_reward_in_step: float = 0.0

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
        # Recovery rate chosen so that at step=recovery_steps the ceiling
        # is ~1% of harsh_penalty: exp(-rate * N) ~ 0.01 => rate = ln(100)/N
        # Link: standard exponential decay, solving for 99% recovery
        self._recovery_rate = np.log(100.0) / max(recovery_steps, 1)

    @staticmethod
    def _compute_ratio(states, info) -> float | None:
        """|net|-to-operator-limit ratio, or None when limit is unavailable.

        The limit is symmetric (grid connection contract), so we use the
        magnitude of ``net_power_kW`` — both excess consumption and excess
        export are equally penalised. Centralised so ``should_terminate``
        and ``get_reward`` never disagree.
        """
        ctxt = states.get("ctxt_operator_max_power_kW")
        if ctxt is None:
            return None
        operator_limit_kW = float(ctxt[0])
        if operator_limit_kW <= 0:
            return None
        net_power_kW = float(info.get("net_power_kW", 0.0))
        return abs(net_power_kW) / operator_limit_kW

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
        max_step = self.weight * self.max_reward_in_step  # always 0.0, kept for clarity

        if info is None:
            logger.warning("OperatorEnergyControlReward: info dict is None, returning 0")
            return 0.0, max_step

        ratio = self._compute_ratio(states, info)
        if ratio is None:
            logger.error("E usage ratio can't be computed")
            return 0.0, 0.0

        # Advance per-episode step counter (resets each episode when the
        # env clears _component_info).
        step = info[_STEP_KEY] = info.get(_STEP_KEY, 0) + 1

        # Terminal breach -- only paid here when the env honours early
        # termination. Otherwise fall through to the over-limit branch.
        if ratio > self.terminate_threshold_pct and info.get("allow_early_termination", False):
            return float(self.weight * self.terminate_penalty), max_step

        # Over-limit (incl. ratio > terminate_threshold_pct when early
        # termination is disabled): flat harsh penalty + mark violation.
        if ratio > 1.0:
            info[_LAST_VIOLATION_KEY] = step
            return float(self.weight * self.harsh_penalty), max_step

        # Safe zone.
        if ratio <= self.soft_threshold_pct:
            return 0.0, max_step

        # Warning zone (soft_threshold_pct < ratio <= 1.0): smooth decay
        # 0 -> ~-1, overridden by the recovery curve when we are still
        # within `recovery_steps` of the last violation.
        steps_since_violation = step - info.get(_LAST_VIOLATION_KEY, -self.recovery_steps)
        if steps_since_violation <= self.recovery_steps:
            reward = self.harsh_penalty * np.exp(-self._recovery_rate * steps_since_violation)
        else:
            t = (ratio - self.soft_threshold_pct) / (1.0 - self.soft_threshold_pct)
            reward = np.exp(-self._DECAY_SCALE * t) - 1.0

        return float(self.weight * reward), max_step


# Register OperatorEnergyControlReward with the component registry
ComponentRegistry.register('reward', OperatorEnergyControlReward)
