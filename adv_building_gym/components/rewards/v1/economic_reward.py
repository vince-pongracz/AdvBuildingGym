import logging
import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# NOTE VP 2026.01.24. : Maybe switch action positive-negative convention... for now it's okay
# But sometimes it is confusing

class EconomicReward(RewardFunction):
    """Economic reward.

    Sign convention (canonical, set by ``EnergyTracker``): positive
    ``net_power_kW`` means EXPORT (feeding into the grid); negative means
    IMPORT (drawing from the grid).

    Reward sign matrix (``E_price`` ∈ [-1, 1] after ABS_MIN_MAX scaling of
    ``baseprice``, which may itself be negative on spot markets):

        consume at positive price → negative reward (cost)
        consume at negative price → positive reward (paid to consume)
        export  at positive price → positive reward (income)
        export  at negative price → negative reward (must pay to dump)

    Formula: ``net_power_kW * E_price / reference_power_kW`` (no leading
    minus). Under the canonical sign convention this matches the matrix
    above directly: ``+net · +price = +`` (export earnings),
    ``-net · +price = -`` (consumption cost).

    Normalisation prefers ``ctxt_operator_max_power_kW`` (published every
    step by ``OperatorEnergyControl``) so the reward magnitude tracks the
    active grid-exchange limit even when ``InfraCombinator`` swaps configs.
    The constructor's ``reference_power_kW`` is a fallback for envs that
    omit ``OperatorEnergyControl``.

    Reads ``net_power_kW`` from the ``info`` dict (published by the
    environment from infrastructure power computations).  ``E_price`` is
    read directly from ``states``.
    """

    def __init__(self, weight: float, 
                reference_power_kW: float = 15.0,
                name: str = "economic_reward") -> None:
        """Initialize EconomicReward.

        Args:
            weight: Reward weight for multi-objective optimization.
            reference_power_kW: Fallback power scale (kW) used when
                ``ctxt_operator_max_power_kW`` is not present in
                ``states`` (i.e. envs without ``OperatorEnergyControl``).
            name: Reward function identifier.
        """
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

    def get_reward(self, actions, state, next_state, info: dict | None = None) -> float:
        # Price the agent observed and acted under (s), not the next row.
        current_energy_price = float(state["s_E_price"][0])

        if info is None:
            logger.warning("EconomicReward: info dict is None, returning 0")
            return 0.0

        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("EconomicReward: missing net_power_kW in info, returning 0")
            return 0.0

        reference_power_kW = self._resolve_reference_power_kW(state)
        # net_power_kW > 0 = EXPORT, < 0 = CONSUME; × signed price gives the
        # docstring's sign matrix (export@+price → +income, consume@+price → -cost).
        raw = net_power_kW * current_energy_price / reference_power_kW

        reward_economic = float(np.clip(raw, -1.0, 1.0))
        return float(self.weight * reward_economic)


# register with ComponentRegistry
ComponentRegistry.register('reward', EconomicReward)
