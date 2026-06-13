"""Rule-based control strategies (heuristic baselines, RL-free).

Deps: numpy + core + components only — keep this package importable
without ray/rllib, stable_baselines3, torch, or pyomo.
"""

from .base import RuleBasedStrategy, in_time_window
from .strategies import (
    DeficitDischargeStrategy,
    DoNothingStrategy,
    PriceMedianStrategy,
    PVSurplusChargeStrategy,
    SelfCoverageStrategy,
)

STRATEGY_REGISTRY: dict[str, type[RuleBasedStrategy]] = {
    cls.name: cls
    for cls in (
        DoNothingStrategy,
        PVSurplusChargeStrategy,
        SelfCoverageStrategy,
        DeficitDischargeStrategy,
        PriceMedianStrategy,
    )
}

__all__ = [
    "RuleBasedStrategy",
    "in_time_window",
    "DoNothingStrategy",
    "PVSurplusChargeStrategy",
    "SelfCoverageStrategy",
    "DeficitDischargeStrategy",
    "PriceMedianStrategy",
    "STRATEGY_REGISTRY",
]
