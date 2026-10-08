"""Rule-based control strategies (heuristic baselines, RL-free).

Deps: numpy + core + components only — keep this package importable
without ray/rllib, stable_baselines3, torch, or pyomo.
"""

from .base import PRICE_STATISTICS, RuleBasedStrategy, in_time_window
from .strategies import (
    SCALED_PRICE_FRACTIONS,
    DeficitDischargeStrategy,
    DoNothingStrategy,
    PriceMeanAutarky,
    PriceMeanStrategy,
    PriceMedianAutarky,
    PriceMedianStrategy,
    PriceThresholdAutarky,
    PriceThresholdStrategy,
    PVSurplusChargeStrategy,
    Autarky,
    make_scaled_price_threshold,
)

# One concrete, registered PriceThresholdStrategy variant per (statistic, strength level).
_SCALED_PRICE_VARIANTS = tuple(
    make_scaled_price_threshold(statistic, fraction)
    for statistic in PRICE_STATISTICS
    for fraction in SCALED_PRICE_FRACTIONS
)

STRATEGY_REGISTRY: dict[str, type[RuleBasedStrategy]] = {
    cls.name: cls
    for cls in (
        DoNothingStrategy,
        PVSurplusChargeStrategy,
        Autarky,
        DeficitDischargeStrategy,
        PriceMedianStrategy,
        PriceMeanStrategy,
        *_SCALED_PRICE_VARIANTS,
        PriceMedianAutarky,
        PriceMeanAutarky,
    )
}

__all__ = [
    "RuleBasedStrategy",
    "in_time_window",
    "PRICE_STATISTICS",
    "DoNothingStrategy",
    "PVSurplusChargeStrategy",
    "Autarky",
    "DeficitDischargeStrategy",
    "PriceThresholdStrategy",
    "PriceMedianStrategy",
    "PriceMeanStrategy",
    "make_scaled_price_threshold",
    "SCALED_PRICE_FRACTIONS",
    "PriceThresholdAutarky",
    "PriceMedianAutarky",
    "PriceMeanAutarky",
    "STRATEGY_REGISTRY",
]
