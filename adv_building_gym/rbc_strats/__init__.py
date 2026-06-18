"""Rule-based control strategies (heuristic baselines, RL-free).

Deps: numpy + core + components only — keep this package importable
without ray/rllib, stable_baselines3, torch, or pyomo.
"""

from .base import RuleBasedStrategy, in_time_window
from .strategies import (
    SCALED_PRICE_MEDIAN_FRACTIONS,
    DeficitDischargeStrategy,
    DoNothingStrategy,
    PriceMedianAutarky,
    PriceMedianStrategy,
    PVSurplusChargeStrategy,
    ScaledPriceMedianStrategy,
    Autarky,
    make_scaled_price_median,
)

# One concrete, registered ScaledPriceMedianStrategy variant per strength level.
_SCALED_PRICE_MEDIAN_VARIANTS = tuple(
    make_scaled_price_median(fraction) for fraction in SCALED_PRICE_MEDIAN_FRACTIONS
)

STRATEGY_REGISTRY: dict[str, type[RuleBasedStrategy]] = {
    cls.name: cls
    for cls in (
        DoNothingStrategy,
        PVSurplusChargeStrategy,
        Autarky,
        DeficitDischargeStrategy,
        PriceMedianStrategy,
        *_SCALED_PRICE_MEDIAN_VARIANTS,
        PriceMedianAutarky,
    )
}

__all__ = [
    "RuleBasedStrategy",
    "in_time_window",
    "DoNothingStrategy",
    "PVSurplusChargeStrategy",
    "Autarky",
    "DeficitDischargeStrategy",
    "PriceMedianStrategy",
    "ScaledPriceMedianStrategy",
    "make_scaled_price_median",
    "SCALED_PRICE_MEDIAN_FRACTIONS",
    "PriceMedianAutarky",
    "STRATEGY_REGISTRY",
]
