import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class EconomicRewardV0(RewardFunction):
    """Dense economic reward (V0) — like ``LongTermEconomicRewardV0`` but per step.

    Canonical sign (EnergyTracker): ``net_power_kW`` > 0 = EXPORT, < 0 = IMPORT.
    Per-step = ``clip(net_power_kW * price_signal / op_max_kW, -1, 1)``, 
    where price_signal rescales the normalised price by the dynamic-max divisor.
    Gives: 
    export at +price → +reward (income)
    consume at +price → −reward (cost)
    export at −price → −reward (paying to dump)
    consume at −price → +reward (paid to consume)
    ``op_max_kW`` from ``ctxt_operator_max_power_kW`` if present, else the ctor fallback.
    """

    _exclude_params = {"_step"}

    def __init__(self, weight: float, reference_power_kW: float = 15.0,
                name: str = "economic_reward_v0") -> None:
        super().__init__(weight, name)
        if reference_power_kW <= 0:
            raise ValueError("reference_power_kW must be positive.")
        self.reference_power_kW = float(reference_power_kW)
        self._step = 0

    def _resolve_reference_power_kW(self, states) -> float:
        ctxt = states.get("ctxt_operator_max_power_kW")
        if ctxt is not None:
            value = float(ctxt[0])
            if value > 0:
                return value
        return self.reference_power_kW

    def on_reset(self, states, info: dict) -> None:
        self._step = 0

    def get_reward(self, actions, state, next_state, info: dict) -> float:
        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("EconomicRewardV0: missing net_power_kW in info, returning 0")
            return 0.0

        # Price (and its scaling ctxt) the agent observed and acted under (s).
        current_energy_price_norm = float(state["s_E_price"][0])

        # rescale price by the data-driven denominator to span more of [-1, 1];
        # ctxt = 1.0 (no-op) when dynamic_max_price_calc is disabled
        dynamic_max = state.get("ctxt_E_price_dynamic_max")
        dynamic_max_ep = state.get("ctxt_E_price_dynamic_max_ep")
        dyn_max = float(dynamic_max[0]) if dynamic_max is not None else 1.0
        dyn_max_ep = float(dynamic_max_ep[0]) if dynamic_max_ep is not None else 1.0

        dyn_max_price_divisor = (dyn_max * 0.7 + dyn_max_ep * 0.3)

        if dyn_max_price_divisor != 0:
            price_signal = current_energy_price_norm / dyn_max_price_divisor
        else:
            price_signal = current_energy_price_norm

        op_max_kW = self._resolve_reference_power_kW(state)

        # Canonical: net > 0 means export, net < 0 means consumption.
        per_step = float(np.clip(net_power_kW * price_signal / op_max_kW, -1.0, 1.0))
        self._step += 1

        return float(self.weight * per_step)


ComponentRegistry.register('reward', EconomicRewardV0)
