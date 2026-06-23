"""Cumulative-price bookkeeping extracted from AdvBuildingGym."""
from typing import Dict

from adv_building_gym._common.constants import SECONDS_PER_HOUR

CT_PER_EUR: float = 100.0


class PriceTracker:
    """Sums per-step electricity cost into a running counter (EUR).

    Positive ``cum_price_EUR`` = money spent (consumption -- positive prices); uses the
    consumption-positive convention, unlike ``EnergyTracker``.
    """

    def __init__(self, control_step_s: int) -> None:
        self.control_step_s = control_step_s
        self.cum_price_EUR: float = 0.0

    def reset(self) -> None:
        self.cum_price_EUR = 0.0

    def add_step_contribution(
        self,
        power_breakdown: Dict[str, tuple[float, float]],
        baseprice_ct_per_kWh: float | None,
    ) -> tuple[float, float]:
        """Add this step's cost. Returns (net_consumption_kW, cost_EUR);
        no-op when ``baseprice_ct_per_kWh`` is None (no price source)."""
        if baseprice_ct_per_kWh is None:
            return 0.0, 0.0

        productions = [val[0] for val in power_breakdown.values()]
        consumptions = [val[1] for val in power_breakdown.values()]

        # Positive when net consumption (cost), negative when net export.
        net_consumption_kW = sum(consumptions) - sum(productions)
        energy_kWh = net_consumption_kW * (self.control_step_s / SECONDS_PER_HOUR)
        cost_EUR = energy_kWh * baseprice_ct_per_kWh / CT_PER_EUR

        self.cum_price_EUR += cost_EUR
        return net_consumption_kW, cost_EUR
