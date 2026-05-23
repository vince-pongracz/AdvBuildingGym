import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class MinimiseEnergyConsumptionRewardV0(RewardFunction):
    """Sparse, episode-aggregated energy throughput reward (V0).

    Per-step value uses the canonical sign convention (``net_power_kW > 0``
    = export, ``< 0`` = consumption). Consumption produces a negative
    per-step value (penalty); export produces a positive per-step value
    (reward):

        ratio    = net_power_kW / op_max_kW
        per_step = clip(ratio, -1, 1)

    Returns ``(0.0, 0.0)`` every step until the natural end of the
    episode (``_step == episode_length``) or until ``info["terminated"]``
    flips True, then flushes:

        reward   = clip(accumulator, -steps_seen, +steps_seen)
        max_step = steps_seen

    Range at flush: ``[-N, +N]``. ``op_max_kW`` resolves from
    ``ctxt_operator_max_power_kW`` when present, else the constructor
    fallback ``reference_power_kW``.
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

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        if info is None:
            logger.warning("MinimiseEnergyConsumptionRewardV0: info dict is None, returning 0")
            return 0.0, 0.0

        episode_length = info.get("episode_length")
        if episode_length is None:
            logger.warning("MinimiseEnergyConsumptionRewardV0: missing episode_length in info, returning 0")
            return 0.0, 0.0
        episode_length = int(episode_length)

        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("MinimiseEnergyConsumptionRewardV0: missing net_power_kW in info, returning 0")
            return 0.0, 0.0

        op_max_kW = self._resolve_reference_power_kW(states)

        # Canonical sign convention (set by EnergyTracker): net_power_kW > 0
        # means EXPORT, < 0 means CONSUMPTION. So `ratio = net / op_max` is
        # negative when consuming → negative per-step reward = penalty
        # (matches the "minimise consumption" intent).
        ratio = net_power_kW / op_max_kW
        # per_step_signed = self.export_scale * ratio if ratio > 0.0 else ratio
        per_step = float(np.clip(ratio, -1.0, 1.0))
        self._accumulated_norm += per_step
        self._step += 1

        terminated = bool(info.get("terminated", False))
        if self._step < episode_length and not terminated:
            return 0.0, 0.0

        steps_seen = self._step
        reward = float(np.clip(self._accumulated_norm, -float(steps_seen), float(steps_seen)))
        return float(self.weight * reward), float(self.weight * steps_seen)


ComponentRegistry.register('reward', MinimiseEnergyConsumptionRewardV0)
