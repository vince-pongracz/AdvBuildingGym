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


# TODO noprio VP 2026.06.16.: This strat does not make any sense
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
        return self._battery_action(self._battery_value(surplus_kW, charging=True))


class Autarky(RuleBasedStrategy):
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
                return self._battery_action(self._battery_value(deficit_kW, charging=False))
            return self.zero_action()
        if surplus_kW > 0.0:
            return self._battery_action(self._battery_value(surplus_kW, charging=True))
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
            return self._battery_action(self._battery_value(surplus_kW, charging=True))
        if surplus_kW < 0.0:
            return self._battery_action(self._battery_value(-surplus_kW, charging=False))
        return self.zero_action()


class PriceMedianStrategy(RuleBasedStrategy):
    """Price arbitrage against the episode median raw price: charge at full
    power while the current price is below the median, discharge at full
    power while above — regardless of renewables or time of day, bounded
    only by the SoC headrooms.

    Uses the raw ``baseprice`` series the env bills with (core/env.py
    _current_baseprice_ct_per_kWh); the price source's s_E_price normalisation
    does not affect billing. Day-ahead prices are public, so reading the episode
    window upfront is a fair heuristic.
    """

    name: ClassVar[str] = "price_median"
    requires_battery: ClassVar[bool] = True
    requires_price: ClassVar[bool] = True

    def __init__(self, env: AdvBuildingGym, *, preserve_start_soc: bool = True):
        super().__init__(env, preserve_start_soc=preserve_start_soc)
        self.median_price: float = 0.0

    def reset(self, obs: dict) -> None:
        super().reset(obs)
        self.median_price = self._episode_median_baseprice()
        logger.debug("Episode median baseprice: %.4f ct/kWh", self.median_price)

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        # baseprice_raw is refreshed by the source's update_state each tick,
        # so it is current (prices are known ahead — no measurement lag).
        price = getattr(self.price_source, "baseprice_raw", None)
        if price is None:
            return self.zero_action()
        if price < self.median_price:
            return self._battery_action(self._battery_value(self._charge_headroom_kW(), charging=True))
        if price > self.median_price:
            return self._battery_action(self._battery_value(self._discharge_headroom_kW(), charging=False))
        return self.zero_action()


# Default charge/discharge strength levels for the preconfigured
# ScaledPriceMedianStrategy variants (one registered strategy each). Full power
# (1.0) is already covered by PriceMedianStrategy; add levels here to register
# more variants.
SCALED_PRICE_MEDIAN_FRACTIONS: tuple[float, ...] = (0.75, 0.5, 0.25)

class ScaledPriceMedianStrategy(RuleBasedStrategy):
    """PriceMedianStrategy with a configurable charge / discharge strength: below
    the episode median price charge at ``charge_fraction`` of the battery's rated
    power, above it discharge at the same fraction of rated power — both still
    bounded by the SoC headrooms (the fraction never pushes past soc_max / the SoC
    floor). ``charge_fraction == 1.0`` reproduces PriceMedianStrategy (full-power
    arbitrage); 0.0 idles.

    This is a template: concrete variants with the fraction baked in are built by
    ``make_scaled_price_median`` and registered as ``price_median_scaled_<f>`` (see
    ``SCALED_PRICE_MEDIAN_FRACTIONS``). Median source and billing caveats match
    PriceMedianStrategy.
    """

    name: ClassVar[str] = "price_median_scaled"
    requires_battery: ClassVar[bool] = True
    requires_price: ClassVar[bool] = True
    # Fraction of rated power (same for charging and discharging); overridden per
    # variant by the factory.
    charge_fraction: ClassVar[float] = 1.0

    def __init__(self, env: AdvBuildingGym, *, preserve_start_soc: bool = True):
        super().__init__(env, preserve_start_soc=preserve_start_soc)
        if not 0.0 <= self.charge_fraction <= 1.0:
            raise ValueError(f"charge_fraction must be in [0, 1], got {self.charge_fraction}")
        self.median_price: float = 0.0

    def reset(self, obs: dict) -> None:
        super().reset(obs)
        self.median_price = self._episode_median_baseprice()
        logger.debug("Episode median baseprice: %.4f ct/kWh", self.median_price)

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        # baseprice_raw is refreshed by the source's update_state each tick,
        # so it is current (prices are known ahead — no measurement lag).
        price = getattr(self.price_source, "baseprice_raw", None)
        if price is None:
            return self.zero_action()
        # Scale the rated power by the fraction; the fraction helpers still clamp
        # to the SoC headroom, so end-SoC >= start-SoC is preserved.
        if price < self.median_price:
            return self._battery_action(self._battery_value_fraction(self.charge_fraction, charging=True))
        if price > self.median_price:
            return self._battery_action(self._battery_value_fraction(self.charge_fraction, charging=False))
        return self.zero_action()


def make_scaled_price_median(charge_fraction: float) -> type[ScaledPriceMedianStrategy]:
    """Build a concrete ScaledPriceMedianStrategy variant with the fraction baked in.

    The single ``charge_fraction`` scales both the charge and discharge legs. The
    registered name is ``price_median_scaled_{charge_fraction}``.
    """
    if not 0.0 <= charge_fraction <= 1.0:
        raise ValueError(f"charge_fraction must be in [0, 1], got {charge_fraction}")
    variant_name = f"price_median_scaled_{charge_fraction}"
    return type(variant_name, (ScaledPriceMedianStrategy,), {
        "name": variant_name,
        "charge_fraction": charge_fraction,
    })


# Number of final episode steps over which PriceMedianAutarky force-drains the
# battery to the SoC floor so no surplus is left at episode end (24 steps = 2 h
# at the standard 300 s control step). Baked into the registered strategy.
PRICE_MEDIAN_AUTARKY_DRAIN_LAST_STEPS: int = 24


class PriceMedianAutarky(RuleBasedStrategy):
    """Hybrid of price-based charging (PriceMedianStrategy) and deficit-based
    discharging (Autarky): charge at full headroom while the price is below the
    episode median; while above the median, discharge only to cover the
    consumption deficit (renewables not covering usage) and idle otherwise, so
    stored energy is saved for later high-price / larger-deficit evening steps.

    Over the final ``drain_last_steps`` steps any energy above the SoC floor is
    discharged regardless of price or deficit, so no surplus remains at the end
    of the episode. Median source and billing caveats match PriceMedianStrategy.
    """

    name: ClassVar[str] = "price_median_autarky"
    requires_battery: ClassVar[bool] = True
    requires_price: ClassVar[bool] = True
    # Final-steps drain window; see PRICE_MEDIAN_AUTARKY_DRAIN_LAST_STEPS.
    drain_last_steps: ClassVar[int] = PRICE_MEDIAN_AUTARKY_DRAIN_LAST_STEPS

    def __init__(self, env: AdvBuildingGym, *, preserve_start_soc: bool = True):
        super().__init__(env, preserve_start_soc=preserve_start_soc)
        if not 0 <= self.drain_last_steps <= self.episode_length:
            raise ValueError(f"drain_last_steps must be in [0, {self.episode_length}], got {self.drain_last_steps}")
        self.median_price: float = 0.0

    def reset(self, obs: dict) -> None:
        super().reset(obs)
        self.median_price = self._episode_median_baseprice()
        logger.debug("Episode median baseprice: %.4f ct/kWh", self.median_price)

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        # End-of-episode drain over the final `drain_last_steps` steps: empty any
        # energy above the SoC floor regardless of price or deficit, so no
        # surplus is left in the battery. `_step` is the 0-based index of the
        # current decision, so episode_length - _step is the steps remaining.
        if self.episode_length - self._step <= self.drain_last_steps:
            headroom_kW = self._discharge_headroom_kW()
            if headroom_kW > 0.0:
                return self._battery_action(self._battery_value(headroom_kW, charging=False))
            return self.zero_action()

        price = getattr(self.price_source, "baseprice_raw", None)
        if price is None:
            return self.zero_action()
        # Below median: charge at full headroom (price-based charge).
        if price < self.median_price:
            return self._battery_action(self._battery_value(self._charge_headroom_kW(), charging=True))
        # Above median: discharge only to cover the deficit; idle while
        # renewables still cover usage so energy is kept for later steps.
        if price > self.median_price:
            deficit_kW = -self._renewable_surplus_kW(last_info)
            if deficit_kW > 0.0:
                return self._battery_action(self._battery_value(deficit_kW, charging=False))
        return self.zero_action()
    
# TODO VP 2026.06.16.: Add a perfectforecast strategy that uses future price, future household consumption and future renewable production to optimally schedule the battery, as an upper bound on the performance of any real strategy.
# So compute the whole day production, the whole day consumption, 