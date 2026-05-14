import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

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
        self._accumulated_norm_price = 0.0

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
        self._accumulated_norm_price = 0.0

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        if info is None:
            logger.warning("LongTermEconomicReward: info dict is None, returning 0")
            return 0.0, 0.0

        episode_length = info.get("episode_length")
        if episode_length is None:
            logger.warning("LongTermEconomicReward: missing episode_length in info, returning 0")
            return 0.0, 0.0
        episode_length = int(episode_length)

        # Net power positive --> export to the grid / production
        # Net power negative --> import from the grid / consumption
        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("LongTermEconomicReward: missing net_power_kW in info, returning 0")
            return 0.0, 0.0

        current_energy_price = float(states["s_E_price"][0])
        reference_power_kW = self._resolve_reference_power_kW(states)
        # NOTE VP 2026.05.07.: how much power do I use from the limit -- the price of it.
        self._accumulated_norm_price += net_power_kW / reference_power_kW * current_energy_price
        # For the early termination scenario -- if the env is configured like that -- 
        # but iter could be fetched from the info dict
        self._step += 1

        # Flush at the natural window boundary OR on early termination.
        # `info["terminated"]` is now set in the env's Phase-1 termination
        # pre-pass *before* any get_reward runs, so it's the single
        # authoritative signal regardless of YAML reward ordering.
        terminated = bool(info.get("terminated", False))
        if self._step < episode_length and not terminated:
            return 0.0, 0.0

        steps = max(self._step, 1)
        window_reward = float(np.clip(self._accumulated_norm_price, -float(steps), 0.0))

        return float(self.weight * window_reward), 0.0


ComponentRegistry.register('reward', LongTermEconomicReward)
