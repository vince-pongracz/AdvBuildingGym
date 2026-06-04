import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


_STEP_KEY = "operator_reward_v0_step"
_LAST_VIOLATION_KEY = "operator_reward_v0_last_violation_step"


class OperatorEnergyControlRewardV0(RewardFunction):
    """Operator energy-limit reward (V0), bounded to ``[-1, 0]`` per step.

    The grid connection limit applies symmetrically in both directions:
    excessive draw AND excessive feed-in are punished equally. The
    decision variable is ``ratio = |net_power_kW| / op_max_kW`` (canonical
    sign convention: ``net > 0`` = export, ``< 0`` = consumption).

    - ``ratio <= soft_threshold_pct``           → ``0``
    - ``soft_threshold_pct < ratio <= 1.0``     → ``exp(-5*t) - 1`` (warning
      zone), with ``t = (ratio - soft) / (1 - soft)``. Decays smoothly
      from ``0`` to ≈ ``-1``.
    - ``ratio > 1.0`` (over-limit)              → flat ``harsh_penalty``
      (default ``-1.0``), followed by ``recovery_steps`` steps where the
      warning-zone reward is overridden by ``harsh_penalty * exp(-rate*k)``
      decaying back towards 0.

    No termination logic and no terminal penalty. Pure-penalty: never
    emits a positive reward, so ``max_reward_in_step = 0``.
    """

    _DECAY_SCALE: float = 5.0
    max_reward_in_step: float = 0.0

    def __init__(self,
                weight: float,
                name: str = "operator_energy_control_reward_v0",
                harsh_penalty: float = -1.0,
                soft_threshold_pct: float = 0.9,
                recovery_steps: int = 3,
                ) -> None:
        super().__init__(weight, name)
        self.harsh_penalty = harsh_penalty
        if not 0.0 < soft_threshold_pct < 1.0:
            raise ValueError("soft_threshold_pct must be in (0, 1)")
        self.soft_threshold_pct = soft_threshold_pct
        self.recovery_steps = recovery_steps
        self._recovery_rate = np.log(100.0) / max(recovery_steps, 1)

    @staticmethod
    def _compute_ratio(states, info) -> float | None:
        # Symmetric grid limit: use |net_power_kW|. Excess consumption AND
        # excess export both trigger penalties.
        ctxt = states.get("ctxt_operator_max_power_kW")
        if ctxt is None:
            return None
        operator_limit_kW = float(ctxt[0])
        if operator_limit_kW <= 0:
            return None
        net_power_kW = float(info.get("net_power_kW", 0.0))
        return abs(net_power_kW) / operator_limit_kW

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        max_step = self.weight * self.max_reward_in_step

        if info is None:
            logger.warning("OperatorEnergyControlRewardV0: info dict is None, returning 0")
            return 0.0, max_step

        ratio = self._compute_ratio(states, info)
        if ratio is None:
            logger.error("E usage ratio can't be computed")
            return 0.0, 0.0

        step = info[_STEP_KEY] = info.get(_STEP_KEY, 0) + 1

        if ratio > 1.0:
            info[_LAST_VIOLATION_KEY] = step
            return float(self.weight * self.harsh_penalty), max_step

        if ratio <= self.soft_threshold_pct:
            return 0.0, max_step

        steps_since_violation = step - info.get(_LAST_VIOLATION_KEY, -self.recovery_steps)
        if steps_since_violation <= self.recovery_steps:
            reward = self.harsh_penalty * np.exp(-self._recovery_rate * steps_since_violation)
        else:
            t = (ratio - self.soft_threshold_pct) / (1.0 - self.soft_threshold_pct)
            reward = np.exp(-self._DECAY_SCALE * t) - 1.0

        return float(self.weight * reward), max_step


ComponentRegistry.register('reward', OperatorEnergyControlRewardV0)
