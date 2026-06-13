"""Concrete rule-based control strategies (flat hierarchy, no concrete-to-concrete inheritance)."""

import logging
from typing import ClassVar

import numpy as np

from adv_building_gym.core.env import AdvBuildingGym
from .base import RuleBasedStrategy, in_time_window

logger = logging.getLogger(__name__)


class DoNothingStrategy(RuleBasedStrategy):
    """Zero action on every action key for the whole episode."""

    name: ClassVar[str] = "do_nothing"

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        return self.zero_action()


class PVSurplusChargeStrategy(RuleBasedStrategy):
    """Charge the battery with the renewable surplus; never discharge.

    Other components consume the renewable production first; only the
    remainder goes to the battery. Surplus the battery cannot absorb
    (power or soc_max limit) is exported to the grid by the env power
    balance and credited in cum_price_EUR (core/_price_tracker.py).
    """

    name: ClassVar[str] = "pv_surplus_charge"
    requires_battery: ClassVar[bool] = True

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        surplus_kW = self._renewable_surplus_kW(last_info)
        if surplus_kW <= 0.0:
            return self.zero_action()
        return self._battery_action(self._charge_value(surplus_kW))


class SelfCoverageStrategy(RuleBasedStrategy):
    """Surplus-charge outside the evening window; inside it, discharge to
    cover the consumption deficit (idle while there is no deficit yet —
    stored energy is kept for later steps in the window)."""

    name: ClassVar[str] = "self_coverage"
    requires_battery: ClassVar[bool] = True

    def __init__(self, env: AdvBuildingGym, *, preserve_start_soc: bool = True,
                evening_start: float = 17.0, evening_end: float = 23.0):
        super().__init__(env, preserve_start_soc=preserve_start_soc)
        for label, hour in (("evening_start", evening_start), ("evening_end", evening_end)):
            if not 0.0 <= hour < 24.0:
                raise ValueError(f"{label} must be in [0, 24), got {hour}")
        if evening_start == evening_end:
            raise ValueError("evening_start and evening_end must differ")
        self.evening_start = evening_start
        self.evening_end = evening_end

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        surplus_kW = self._renewable_surplus_kW(last_info)
        if in_time_window(self._hour_of_day(), self.evening_start, self.evening_end):
            deficit_kW = -surplus_kW
            if deficit_kW > 0.0:
                return self._battery_action(self._discharge_value(deficit_kW))
            return self.zero_action()
        if surplus_kW > 0.0:
            return self._battery_action(self._charge_value(surplus_kW))
        return self.zero_action()


class DeficitDischargeStrategy(RuleBasedStrategy):
    """Surplus-charge whenever renewables exceed consumption; discharge to
    cover the deficit whenever consumption exceeds renewables, at any time
    of day. The SoC floor guarantees end-SoC >= start-SoC."""

    name: ClassVar[str] = "deficit_discharge"
    requires_battery: ClassVar[bool] = True

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        surplus_kW = self._renewable_surplus_kW(last_info)
        if surplus_kW > 0.0:
            return self._battery_action(self._charge_value(surplus_kW))
        if surplus_kW < 0.0:
            return self._battery_action(self._discharge_value(-surplus_kW))
        return self.zero_action()


class PriceMedianStrategy(RuleBasedStrategy):
    """Price arbitrage against the episode median raw price: charge at full
    power while the current price is below the median, discharge at full
    power while above — regardless of renewables or time of day, bounded
    only by the SoC headrooms.

    Uses the raw ``baseprice`` series the env bills with (core/env.py
    _current_baseprice_ct_per_kWh); the evening observation boost in
    EnergyPriceDataSource does not affect billing. Day-ahead prices are
    public, so reading the episode window upfront is a fair heuristic.
    """

    name: ClassVar[str] = "price_median"
    requires_battery: ClassVar[bool] = True
    requires_price: ClassVar[bool] = True

    def __init__(self, env: AdvBuildingGym, *, preserve_start_soc: bool = True):
        super().__init__(env, preserve_start_soc=preserve_start_soc)
        self.median_price: float = 0.0

    def reset(self, obs: dict) -> None:
        super().reset(obs)
        ts = self.price_source.ts
        if ts is None or "baseprice" not in ts.columns:
            raise RuntimeError(f"Strategy '{self.name}': price source has no 'baseprice' data loaded.")
        # Same episode window slice the source itself uses in its reset()
        start = self.price_source.row_offset
        end = min(start + self.episode_length, len(ts))
        self.median_price = float(np.median(ts["baseprice"].iloc[start:end]))
        logger.debug("Episode median baseprice: %.4f ct/kWh", self.median_price)

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        # baseprice_raw is refreshed by the source's update_state each tick,
        # so it is current (prices are known ahead — no measurement lag).
        price = getattr(self.price_source, "baseprice_raw", None)
        if price is None:
            return self.zero_action()
        if price < self.median_price:
            return self._battery_action(self._charge_value(self._charge_headroom_kW()))
        if price > self.median_price:
            return self._battery_action(self._discharge_value(self._discharge_headroom_kW()))
        return self.zero_action()
