import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class LongTermEconomicRewardV0(RewardFunction):
    """Sparse, episode-aggregated economic reward (V0).

    Per step accumulates ``clip(net_power_kW * E_price / op_max_kW, -1, 1)``
    into an internal counter, using the canonical sign convention
    (``net_power_kW > 0`` = export, ``< 0`` = consumption). Returns
    ``(0.0, 0.0)`` every step until the natural end of the episode
    (``_step == episode_length``) or until ``info["terminated"]`` flips
    True, then flushes the accumulator:

        reward   = clip(accumulator, -steps_seen, +steps_seen)
        max_step = steps_seen

    Sign matrix per step (matches ``EconomicRewardV0``):

        export at +price → +reward (income)
        consume at +price → −reward (cost)

    Range at flush: ``[-N, +N]`` where ``N = steps_seen``. Per-step value
    is in ``[-1, 1]`` so the magnitude is commensurate with the cumulative
    return of a dense ``EconomicRewardV0`` over the same window.
    """

    _exclude_params = {"_step", "_accumulated_norm"}

    def __init__(self, weight: float, reference_power_kW: float = 15.0,
                name: str = "long_term_economic_reward_v0") -> None:
        super().__init__(weight, name)
        if reference_power_kW <= 0:
            raise ValueError("reference_power_kW must be positive.")
        self.reference_power_kW = float(reference_power_kW)
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
            logger.warning("LongTermEconomicRewardV0: info dict is None, returning 0")
            return 0.0, 0.0

        episode_length = info.get("episode_length")
        if episode_length is None:
            logger.warning("LongTermEconomicRewardV0: missing episode_length in info, returning 0")
            return 0.0, 0.0
        episode_length = int(episode_length)

        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("LongTermEconomicRewardV0: missing net_power_kW in info, returning 0")
            return 0.0, 0.0

        current_energy_price_norm = float(states["s_E_price"][0]) 
        
        # Rescale the price by the data-driven denominator so per-step values
        # span more of [-1, 1]; ctxt is 1.0 (no-op) when EnergyPriceDataSource's
        # dynamic_max_price_calc is disabled.
        dynamic_max = states.get("ctxt_E_price_dynamic_max")
        dynamic_max_ep = states.get("ctxt_E_price_dynamic_max_ep")
        dyn_max = float(dynamic_max[0]) if dynamic_max is not None else 1.0
        dyn_max_ep = float(dynamic_max_ep[0]) if dynamic_max_ep is not None else 1.0
        
        dyn_max_price_divisor = (dyn_max * 0.7 + dyn_max_ep * 0.3)
        
        # During the evening peak, the price signal is often very low due to the high max price, 
        # which makes it hard for the agent to learn. So we scale up the price signal by 2 during this period.
        # 18:00 to 21:00, when the price is usually high 
        # --> make it even more higher to discourage consumption during this period.
        if self._step > 216 and self._step < 252: 
            current_energy_price_norm *=2
        
        if dyn_max_price_divisor != 0:
            price_signal = current_energy_price_norm / dyn_max_price_divisor
        else:
            price_signal = current_energy_price_norm

        op_max_kW = self._resolve_reference_power_kW(states)

        # Canonical: net > 0 means export, net < 0 means consumption.
        per_step = float(np.clip(net_power_kW * price_signal / op_max_kW, -1.0, 1.0))
        # per_step = float(net_power_kW * price_signal)
        self._accumulated_norm += per_step
        self._step += 1

        terminated = bool(info.get("terminated", False))
        if self._step < episode_length and not terminated:
            return 0.0, 0.0

        steps_seen = self._step
        # reward = float(np.clip(self._accumulated_norm, -float(steps_seen), float(steps_seen)))
        reward = self._accumulated_norm
        return float(self.weight * reward), float(self.weight * steps_seen)


ComponentRegistry.register('reward', LongTermEconomicRewardV0)
