"""Concrete rule-based control strategies (flat hierarchy, no concrete-to-concrete inheritance)."""

import logging
from typing import ClassVar

import numpy as np

from adv_building_gym.core.env import AdvBuildingGym
from .base import PRICE_STATISTICS, RuleBasedStrategy, in_time_window

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


class PriceThresholdStrategy(RuleBasedStrategy):
    """Price arbitrage against an episode-wide reference price: charge while the
    current price is below the reference, discharge while above — regardless of
    renewables or time of day, bounded only by the SoC headrooms.

    ``price_statistic`` picks the aggregator that collapses the episode price
    window into the reference (``base.PRICE_STATISTICS``); ``charge_fraction``
    scales both the charge and the discharge leg to that fraction of the
    battery's rated power (1.0 = full power, 0.0 idles). The SoC headroom caps
    apply on top, so end-SoC >= start-SoC still holds.

    Uses the raw ``baseprice`` (``get_raw_values()["raw_E_price"]``) the env bills with;
    the price source's s_E_price normalisation does not affect billing. Day-ahead prices
    are public, so reading the episode window upfront is a fair heuristic.

    This is a template: the registered strategies are the concrete subclasses
    below and the variants built by ``make_scaled_price_threshold``.
    """

    name: ClassVar[str] = "price_threshold"
    requires_battery: ClassVar[bool] = True
    requires_price: ClassVar[bool] = True
    price_statistic: ClassVar[str] = "median"
    # Fraction of rated power (same for charging and discharging).
    charge_fraction: ClassVar[float] = 1.0

    def __init__(self, env: AdvBuildingGym, *, preserve_start_soc: bool = True):
        super().__init__(env, preserve_start_soc=preserve_start_soc)
        if not 0.0 <= self.charge_fraction <= 1.0:
            raise ValueError(f"charge_fraction must be in [0, 1], got {self.charge_fraction}")
        self.reference_price: float = 0.0

    def reset(self, obs: dict) -> None:
        super().reset(obs)
        self.reference_price = self._episode_reference_baseprice(self.price_statistic)
        logger.debug("Episode %s baseprice: %.4f ct/kWh", self.price_statistic, self.reference_price)

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        # current price (known ahead — no measurement lag)
        price = self._current_baseprice()
        if price is None:
            return self.zero_action()
        # Scale the rated power by the fraction; the fraction helper still clamps
        # to the SoC headroom, so end-SoC >= start-SoC is preserved.
        if price < self.reference_price:
            return self._battery_action(self._battery_value_fraction(self.charge_fraction, charging=True))
        if price > self.reference_price:
            return self._battery_action(self._battery_value_fraction(self.charge_fraction, charging=False))
        return self.zero_action()


class PriceMedianStrategy(PriceThresholdStrategy):
    """Full-power price arbitrage against the episode MEDIAN raw price."""

    name: ClassVar[str] = "price_median"
    price_statistic: ClassVar[str] = "median"


class PriceMeanStrategy(PriceThresholdStrategy):
    """Full-power price arbitrage against the episode MEAN raw price.

    Unlike the median, the mean does not split the day into equally many
    charge/discharge steps: on a skewed price day it moves the threshold towards
    the tail, trading charge steps for discharge steps (or vice versa) relative
    to PriceMedianStrategy.
    """

    name: ClassVar[str] = "price_mean"
    price_statistic: ClassVar[str] = "mean"


# Default charge/discharge strength levels for the preconfigured scaled
# price-threshold variants (one registered strategy per statistic and level).
# Full power (1.0) is already covered by PriceMedianStrategy / PriceMeanStrategy;
# add levels here to register more variants.
SCALED_PRICE_FRACTIONS: tuple[float, ...] = (0.75, 0.5, 0.25)


def make_scaled_price_threshold(price_statistic: str, charge_fraction: float) -> type[PriceThresholdStrategy]:
    """Build a concrete PriceThresholdStrategy variant with statistic and fraction baked in.

    The single ``charge_fraction`` scales both the charge and discharge legs. The
    registered name is ``price_{price_statistic}_scaled_{charge_fraction}``.
    """
    if price_statistic not in PRICE_STATISTICS:
        raise ValueError(f"Unknown price statistic '{price_statistic}'; expected one of {sorted(PRICE_STATISTICS)}")
    if not 0.0 <= charge_fraction <= 1.0:
        raise ValueError(f"charge_fraction must be in [0, 1], got {charge_fraction}")
    variant_name = f"price_{price_statistic}_scaled_{charge_fraction}"
    return type(variant_name, (PriceThresholdStrategy,), {
        "name": variant_name,
        "price_statistic": price_statistic,
        "charge_fraction": charge_fraction,
    })


# Number of final episode steps over which the price-threshold autarky strategies
# force-drain the battery to the SoC floor so no surplus is left at episode end
# (24 steps = 2 h at the standard 300 s control step).
PRICE_THRESHOLD_AUTARKY_DRAIN_LAST_STEPS: int = 24


class PriceThresholdAutarky(RuleBasedStrategy):
    """Hybrid of price-based charging (PriceThresholdStrategy) and deficit-based
    discharging (Autarky): charge at full headroom while the price is below the
    episode reference price; while above it, discharge only to cover the
    consumption deficit (renewables not covering usage) and idle otherwise, so
    stored energy is saved for later high-price / larger-deficit evening steps.

    Over the final ``drain_last_steps`` steps any energy above the SoC floor is
    discharged regardless of price or deficit, so no surplus remains at the end
    of the episode. ``price_statistic`` picks the reference aggregator; price
    source and billing caveats match PriceThresholdStrategy.

    This is a template: the registered strategies are the concrete subclasses below.
    """

    name: ClassVar[str] = "price_threshold_autarky"
    requires_battery: ClassVar[bool] = True
    requires_price: ClassVar[bool] = True
    price_statistic: ClassVar[str] = "median"
    # Final-steps drain window; see PRICE_THRESHOLD_AUTARKY_DRAIN_LAST_STEPS.
    drain_last_steps: ClassVar[int] = PRICE_THRESHOLD_AUTARKY_DRAIN_LAST_STEPS

    def __init__(self, env: AdvBuildingGym, *, preserve_start_soc: bool = True):
        super().__init__(env, preserve_start_soc=preserve_start_soc)
        if not 0 <= self.drain_last_steps <= self.episode_length:
            raise ValueError(f"drain_last_steps must be in [0, {self.episode_length}], got {self.drain_last_steps}")
        self.reference_price: float = 0.0

    def reset(self, obs: dict) -> None:
        super().reset(obs)
        self.reference_price = self._episode_reference_baseprice(self.price_statistic)
        logger.debug("Episode %s baseprice: %.4f ct/kWh", self.price_statistic, self.reference_price)

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

        price = self._current_baseprice()
        if price is None:
            return self.zero_action()
        # Below the reference: charge at full headroom (price-based charge).
        if price < self.reference_price:
            return self._battery_action(self._battery_value(self._charge_headroom_kW(), charging=True))
        # Above the reference: discharge only to cover the deficit; idle while
        # renewables still cover usage so energy is kept for later steps.
        if price > self.reference_price:
            deficit_kW = -self._renewable_surplus_kW(last_info)
            if deficit_kW > 0.0:
                return self._battery_action(self._battery_value(deficit_kW, charging=False))
        return self.zero_action()


class PriceMedianAutarky(PriceThresholdAutarky):
    """PriceThresholdAutarky referenced against the episode MEDIAN raw price."""

    name: ClassVar[str] = "price_median_autarky"
    price_statistic: ClassVar[str] = "median"


class PriceMeanAutarky(PriceThresholdAutarky):
    """PriceThresholdAutarky referenced against the episode MEAN raw price."""

    name: ClassVar[str] = "price_mean_autarky"
    price_statistic: ClassVar[str] = "mean"


# TODO VP 2026.06.16.: Add a perfectforecast strategy that uses future price, future household consumption and future renewable production to optimally schedule the battery, as an upper bound on the performance of any real strategy.
# So compute the whole day production, the whole day consumption, 