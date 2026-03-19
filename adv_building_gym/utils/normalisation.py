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
        # TODO VP 2026.03.18. : Log if div by zero would have happened
        case Normalisation.ABS_MIN_MAX_SCALING:
            # norm = val / max(|min|, |max|), maps to [-1, 1]
            abs_max = max(abs(series.min()), abs(series.max()))
            return series / abs_max if abs_max != 0 else series * 0.0
        case Normalisation.MAX_ABS_SCALING:
            max_abs = series.abs().max()
            return series / max_abs if max_abs != 0 else series * 0.0
        case Normalisation.MIN_MAX_SCALING:
            range_ = series.max() - series.min()
            return (series - series.min()) / range_ if range_ != 0 else series * 0.0
        case Normalisation.STANDARDISATION:
            std = series.std()
            return (series - series.mean()) / std if std != 0 else series * 0.0
        case None:
            return series
        case _:
            raise ValueError(f"Unknown normalisation method: {method}")
