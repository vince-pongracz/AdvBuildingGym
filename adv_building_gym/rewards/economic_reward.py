import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# NOTE VP 2026.01.24. : Maybe switch action positive-negative convention... for now it's okay
# But sometimes it is confusing

class EconomicReward(RewardFunction):
    """Economic-based reward function.

    Action convention: positive ``net_power_kW`` means drawing from the
    grid (consumption); negative means feeding into the grid (export).

    Reward sign matrix (``E_price`` ∈ [-1, 1] after ABS_MIN_MAX scaling of
    ``baseprice``, which may itself be negative on spot markets):

        consume at positive price → negative reward (cost)
        consume at negative price → positive reward (paid to consume)
        export  at positive price → positive reward (income)
        export  at negative price → negative reward (must pay to dump)

    Normalisation uses a single, topology-independent ``reference_power_kW``
    so the reward magnitude does not depend on how much infrastructure the
    environment happens to contain.  Keep it aligned with the grid-limit
    parameter used by ``OperatorEnergyControlReward.max_power_kW`` so both
    rewards speak the same "typical grid exchange" scale.

    Reads ``net_power_kW`` from the ``info`` dict (published by the
    environment from infrastructure power computations).  ``E_price`` is
    read directly from ``states``.
    """

    def __init__(self, weight: float, reference_power_kW: float,
                name: str = "economic_reward",
                export_bonus: float = 1.0) -> None:
        """Initialize EconomicReward.

        Args:
            weight: Reward weight for multi-objective optimization.
            reference_power_kW: Power scale (kW) used to normalise the
                reward into [-1, 1].  Should match the grid-exchange limit
                used by ``OperatorEnergyControlReward.max_power_kW``.
            name: Reward function identifier.
            export_bonus: Multiplier applied when the building is exporting
                to the grid (``net_power_kW < 0``).  Values > 1 make
                selling energy more attractive relative to buying it.
                Applied by the sign of ``net_power_kW``, not the sign of
                the reward, so it behaves consistently under negative
                spot prices.
        """
        super().__init__(weight, name)
        if reference_power_kW <= 0:
            raise ValueError("reference_power_kW must be positive.")
        self.reference_power_kW = float(reference_power_kW)
        self.export_bonus = export_bonus

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        max_step = self.weight * self.max_reward

        current_energy_price = float(states["s_E_price"][0])

        if info is None:
            logger.warning("EconomicReward: info dict is None, returning 0")
            return 0.0, max_step

        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("EconomicReward: missing net_power_kW in info, returning 0")
            return 0.0, max_step

        # Negative sign: consumption → negative reward (cost); production → positive reward (income)
        raw = -net_power_kW * current_energy_price / self.reference_power_kW

        # Export side boost — keyed on the physical direction of power flow,
        # not on the reward sign, so negative prices don't flip the meaning.
        if net_power_kW < 0 and self.export_bonus != 1.0:
            raw *= self.export_bonus

        reward_economic = float(np.clip(raw, -1.0, 1.0))
        return float(self.weight * reward_economic), max_step


# Register EconomicReward with the component registry
ComponentRegistry.register('reward', EconomicReward)
