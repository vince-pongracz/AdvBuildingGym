"""Shared helpers for per-episode uniform parameter sampling with exclusions.

Used by the sampling infrastructure wrappers (``BatteryLinearWrapper``,
``SolarPanelWrapper``): uniform draw from a [low, high] range, rounded to the
1-decimal grid, redrawn while the value sits in an exclusion list (held-out
sizes for eval generalisation).
"""

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Defensive cap for the rejection-sampling loop; unreachable in practice because
# the wrappers validate that the exclusions never cover the whole 1-decimal grid.
_MAX_SAMPLE_ATTEMPTS = 1000


def _as_tenths(value: float) -> int:
    """Integer tenths of a 1-decimal value (17.3 → 173) — float-equality-safe comparisons."""
    return int(round(value * 10))


def _validate_range(param_name: str, value_range: Sequence[float]) -> list[float]:
    """Validate a [low, high] sampling range and return it as a float list."""
    if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
        raise ValueError(f"{param_name} must be a 2-item list [low, high], got {value_range!r}.")
    low, high = float(value_range[0]), float(value_range[1])
    if low < 0.0:
        raise ValueError(f"{param_name} bounds must be non-negative, got {value_range!r}.")
    if low > high:
        raise ValueError(f"{param_name} must satisfy low <= high, got {value_range!r}.")
    return [low, high]


def _validate_exclusions(param_name: str, value_range: list[float],
                        excluded_values: Sequence[float] | None) -> frozenset[int]:
    """Excluded values as a tenth-integer set; reject exclusions covering the whole range grid.

    Rounded draws live on the 1-decimal grid between round(low, 1) and
    round(high, 1) — if every grid point is excluded, sampling could never
    terminate, so fail fast here instead of looping at reset time.
    """
    excluded_tenths = frozenset(_as_tenths(float(v)) for v in (excluded_values or []))
    grid = set(range(_as_tenths(round(value_range[0], 1)), _as_tenths(round(value_range[1], 1)) + 1))
    if grid <= excluded_tenths:
        raise ValueError(
            f"{param_name} excludes every 1-decimal value in range {value_range} — nothing left to sample."
        )
    return excluded_tenths


def _draw_rounded_uniform_excluding(rng: np.random.Generator, low: float, high: float,
                                    excluded_tenths: frozenset[int]) -> float:
    """Uniform draw from [low, high] rounded to 1 decimal, redrawn while excluded."""
    for _ in range(_MAX_SAMPLE_ATTEMPTS):
        value = round(float(rng.uniform(low, high)), 1)
        if _as_tenths(value) not in excluded_tenths:
            return value
    raise RuntimeError(
        f"No acceptable sample from [{low}, {high}] after {_MAX_SAMPLE_ATTEMPTS} attempts "
        f"(excluded tenths: {sorted(excluded_tenths)})."
    )
