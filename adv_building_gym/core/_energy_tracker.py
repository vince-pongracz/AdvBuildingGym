"""Cumulative-energy bookkeeping extracted from AdvBuildingGym."""
from typing import Dict

from adv_building_gym._common.constants import SECONDS_PER_HOUR


class EnergyTracker:
    """Sums per-infra power into a running cumulative-energy counter (kWh)."""

    def __init__(self, control_step_s: int) -> None:
        self.control_step_s = control_step_s
        self.cum_E_kWh: float = 0.0

    def reset(self) -> None:
        self.cum_E_kWh = 0.0

    def add_step_E_contrib(self, power_breakdown: Dict[str, tuple[float, float]]) -> tuple[float, float]:
        """Add this step's energy contribution. Returns (total_power_kW, energy_kWh)."""
        productions = [val[0] for val in power_breakdown.values()]
        consumptions = [val[1] for val in power_breakdown.values()]
        
        # Positive if energy export, negative if energy usage
        total_power_balance_kW = sum(productions) - sum(consumptions)
        energy_kWh = total_power_balance_kW * (self.control_step_s / SECONDS_PER_HOUR)
        
        self.cum_E_kWh += energy_kWh
        return total_power_balance_kW, energy_kWh
