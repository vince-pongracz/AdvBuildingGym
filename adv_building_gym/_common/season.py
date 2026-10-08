"""Meteorological season definitions (Northern hemisphere), shared across layers.

Link: https://www.ncei.noaa.gov/news/meteorological-versus-astronomical-seasons
"""

from __future__ import annotations

from typing import Final

# Sentinel meaning "no season restriction".
ALL_SEASONS: Final[str] = "all"

# Each base season is a whole calendar quarter; composites are unions of them.
SEASON_MONTHS: Final[dict[str, frozenset[int]]] = {
    "winter": frozenset({12, 1, 2}),
    "spring": frozenset({3, 4, 5}),
    "summer": frozenset({6, 7, 8}),
    "autumn": frozenset({9, 10, 11}),
}
SEASON_MONTHS["spring_autumn"] = SEASON_MONTHS["spring"] | SEASON_MONTHS["autumn"]

_BASE_SEASONS: Final[tuple[str, ...]] = ("winter", "spring", "summer", "autumn")


def season_for_month(month: int) -> str:
    """Meteorological season name (winter/spring/summer/autumn) for a 1-12 month."""
    for season in _BASE_SEASONS:
        if month in SEASON_MONTHS[season]:
            return season
    raise ValueError(f"month must be in 1-12, got {month}")
