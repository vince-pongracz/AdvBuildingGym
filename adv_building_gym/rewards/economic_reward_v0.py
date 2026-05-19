import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class EconomicRewardV0(RewardFunction):
    """Economic-based reward function (V0).

    Identical to ``EconomicReward``: per-step clipped to ``[-1, 1]``.

    Sign convention (canonical, set by ``EnergyTracker``): positive
    ``net_power_kW`` means EXPORT (feeding into the grid); negative means
    IMPORT (drawing from the grid).

    Reward = ``net_power_kW * E_price / reference_power_kW``,
    clipped to ``[-1, 1]``. Under the canonical convention this gives:

        export at +price → +reward (income)
        consume at +price → −reward (cost)
        export at −price → −reward (paying to dump)
        consume at −price → +reward (paid to consume)

    ``reference_power_kW`` resolves from ``ctxt_operator_max_power_kW``
    when present, else the constructor fallback.
    """

    def __init__(self, weight: float, reference_power_kW: float = 15.0,
                name: str = "economic_reward_v0") -> None:
        super().__init__(weight, name)
        if reference_power_kW <= 0:
            raise ValueError("reference_power_kW must be positive.")
        self.reference_power_kW = float(reference_power_kW)

    def _resolve_reference_power_kW(self, states) -> float:
        ctxt = states.get("ctxt_operator_max_power_kW")
        if ctxt is not None:
            value = float(ctxt[0])
            if value > 0:
                return value
        return self.reference_power_kW

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        max_reward_per_step = self.weight * self.max_reward_in_step

        current_energy_price = float(states["s_E_price"][0])

        if info is None:
            logger.warning("EconomicRewardV0: info dict is None, returning 0")
            return 0.0, max_reward_per_step

        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("EconomicRewardV0: missing net_power_kW in info, returning 0")
            return 0.0, max_reward_per_step

        reference_power_kW = self._resolve_reference_power_kW(states)
        # Canonical: net > 0 means export, net < 0 means consumption.
        raw = net_power_kW * current_energy_price / reference_power_kW

        reward_economic = float(np.clip(raw, -1.0, 1.0))
        return float(self.weight * reward_economic), max_reward_per_step


ComponentRegistry.register('reward', EconomicRewardV0)
