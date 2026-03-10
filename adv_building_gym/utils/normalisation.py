"""Reusable normalisation utilities for pandas Series."""

from enum import Enum

import pandas as pd


class Normalisation(Enum):
    """Normalisation types"""
    MAX_ABS_SCALING = "max_abs"
    ABS_MIN_MAX_SCALING = "abs_min_max"
    MIN_MAX_SCALING = "min_max"
    STANDARDISATION = "std"

    @classmethod
    def init(cls, value: "Normalisation | str | None") -> "Normalisation | None":
        """Convert a string to a Normalisation enum member, pass through enum/None unchanged."""
        if value is None or isinstance(value, cls):
            return value
        return cls(value)


def normalise_series(series: pd.Series, method: Normalisation | None) -> pd.Series:
    """Normalise a pandas Series using the given method.

    Args:
        series: Input data.
        method: Normalisation strategy. None returns the series unchanged.

    Returns:
        Normalised pandas Series.
    """
    match method:
        case Normalisation.ABS_MIN_MAX_SCALING:
            # norm = val / max(|min|, |max|), maps to [-1, 1]
            abs_max = max(abs(series.min()), abs(series.max()))
            return series / abs_max if abs_max != 0 else series * 0.0
        case Normalisation.MAX_ABS_SCALING:
            return series / series.abs().max()
        case Normalisation.MIN_MAX_SCALING:
            return (series - series.min()) / (series.max() - series.min())
        case Normalisation.STANDARDISATION:
            return (series - series.mean()) / series.std()
        case None:
            return series
        case _:
            raise ValueError(f"Unknown normalisation method: {method}")
