import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)

class LongTermEconomicReward(RewardFunction):
    """Sparse, period-aggregated economic reward.
    """

    def __init__(self, weight: float, reference_power_kW: float = 15.0,
                name: str = "long_term_economic_reward") -> None:
        """Initialize LongTermEconomicReward.

        Args:
            weight: Reward weight for multi-objective optimization.
            reference_power_kW: Fallback power scale (kW) used when
                ``ctxt_operator_max_power_kW`` is not present in
                ``states``. Otherwise that ctxt value is preferred so
                this reward auto-tracks the active grid-exchange limit
                (matching ``EconomicReward``).
            name: Reward function identifier.
        """
        super().__init__(weight, name)
        if reference_power_kW <= 0:
            raise ValueError("reference_power_kW must be positive.")

        self.reference_power_kW = float(reference_power_kW)
        self._step = 0
        self._accumulated_usage_price_norm = 0.0
        self._sum_net_power_kW = 0.0

    _exclude_params = {"_step_in_window", "_accum"}

    def _resolve_reference_power_kW(self, states) -> float:
        ctxt = states.get("ctxt_operator_max_power_kW")
        if ctxt is not None:
            value = float(ctxt[0])
            if value > 0:
                return value
        return self.reference_power_kW
    
    def on_reset(self, states, info: dict | None = None) -> None:
        self._step = 0
        self._accumulated_usage_price_norm = 0.0
        self._sum_net_power_kW = 0.0

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        if info is None:
            logger.warning("LongTermEconomicReward: info dict is None, returning 0")
            return 0.0

        episode_length = info.get("episode_length")
        if episode_length is None:
            logger.warning("LongTermEconomicReward: missing episode_length in info, returning 0")
            return 0.0
        episode_length = int(episode_length)

        # Net power positive --> export to the grid / production
        # Net power negative --> import from the grid / consumption
        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("LongTermEconomicReward: missing net_power_kW in info, returning 0")
            return 0.0

        current_energy_price_norm = float(state["s_E_price"][0])  # observed price (s)
        # _accumulated_usage_price_norm: positive -- we produce, we get paid
        # _accumulated_usage_price_norm: negative -- we consume, we pay
        self._accumulated_usage_price_norm += net_power_kW * current_energy_price_norm
        self._sum_net_power_kW += net_power_kW
        # step counter (for early termination; could also read iter from info)
        self._step += 1

        # Flush at the window boundary OR on early termination. info["terminated"] is
        # set in the env's termination pre-pass, so it's authoritative regardless of reward order.
        terminated = bool(info.get("terminated", False))
        if self._step < episode_length and not terminated:
            return 0.0

        return float(self.weight * self._accumulated_usage_price_norm)


ComponentRegistry.register('reward', LongTermEconomicReward)
