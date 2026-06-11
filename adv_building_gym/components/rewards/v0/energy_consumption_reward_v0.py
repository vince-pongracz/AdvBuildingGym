import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class MinimiseEnergyConsumptionRewardV0(RewardFunction):
    """Sparse, episode-aggregated energy throughput reward (V0).

    Per step (canonical sign: net>0 export, <0 consume): per_step = clip(net_power_kW/op_max_kW, -1, 1)
    — consume → penalty, export → reward. Returns 0 until episode end
    (``_step == episode_length``) or ``info["terminated"]``, then flushes
    clip(accumulator, ±steps_seen).
    ``op_max_kW`` from ``ctxt_operator_max_power_kW`` if present, else ``reference_power_kW``.
    """

    _exclude_params = {"_step", "_accumulated_norm"}

    def __init__(self, weight: float, reference_power_kW: float = 20.0,
                export_scale: float = 0.3,
                name: str = "E_consumption_reward_v0") -> None:
        super().__init__(weight, name)
        if reference_power_kW <= 0:
            raise ValueError("reference_power_kW must be positive.")
        if export_scale < 0.0:
            raise ValueError("export_scale must be non-negative.")
        self.reference_power_kW = float(reference_power_kW)
        self.export_scale = float(export_scale)
        self._step = 0
        self._accumulated_norm = 0.0

    def _resolve_reference_power_kW(self, states) -> float:
        ctxt = states.get("ctxt_operator_max_power_kW")
        if ctxt is not None:
            value = float(ctxt[0])
            if value > 0:
                return value
        return self.reference_power_kW

    def on_reset(self, states, info: dict | None = None) -> None:
        self._step = 0
        self._accumulated_norm = 0.0

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        if info is None:
            logger.warning("MinimiseEnergyConsumptionRewardV0: info dict is None, returning 0")
            return 0.0

        episode_length = info.get("episode_length")
        if episode_length is None:
            logger.warning("MinimiseEnergyConsumptionRewardV0: missing episode_length in info, returning 0")
            return 0.0
        episode_length = int(episode_length)

        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("MinimiseEnergyConsumptionRewardV0: missing net_power_kW in info, returning 0")
            return 0.0

        # Operator limit is a static per-episode ctxt; read it from observed s.
        op_max_kW = self._resolve_reference_power_kW(state)

        # Canonical sign (EnergyTracker): net>0 EXPORT, <0 CONSUME, so ratio is
        # negative when consuming → penalty (matches "minimise consumption").
        ratio = net_power_kW / op_max_kW
        # per_step_signed = self.export_scale * ratio if ratio > 0.0 else ratio
        per_step = float(np.clip(ratio, -1.0, 1.0))
        self._accumulated_norm += per_step
        self._step += 1

        terminated = bool(info.get("terminated", False))
        if self._step < episode_length and not terminated:
            return 0.0

        steps_seen = self._step
        reward = float(np.clip(self._accumulated_norm, -float(steps_seen), float(steps_seen)))
        return float(self.weight * reward)


ComponentRegistry.register('reward', MinimiseEnergyConsumptionRewardV0)
