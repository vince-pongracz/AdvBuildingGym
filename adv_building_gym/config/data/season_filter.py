"""Season-based filtering of selectable episode start days.

Restricts the day-of-year slots a DataCombinator may start an episode on to a
single meteorological season, so train/eval can be scoped to e.g. winter without
touching the day-selection cadence.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from adv_building_gym._common.season import ALL_SEASONS, SEASON_MONTHS as _SEASON_MONTHS

logger = logging.getLogger(__name__)


class SeasonFilter:
    """Restricts selectable episode start days to one meteorological season.

    A day-slot index ``d`` maps to the calendar date ``Jan 1 of data_start_year
    + d days`` -- the same arithmetic DataCombinator uses for its day date -- and
    the filter keeps only slots whose month falls in the configured season.

    Args:
        season: One of ``winter``/``spring``/``summer``/``autumn``/
            ``spring_autumn``, or ``"all"`` (default) for no filtering.
    """

    def __init__(self, season: str = ALL_SEASONS) -> None:
        season = (season or ALL_SEASONS).lower()
        if season != ALL_SEASONS and season not in _SEASON_MONTHS:
            raise ValueError(
                f"Unknown season {season!r}; expected {ALL_SEASONS!r} or one of "
                f"{sorted(_SEASON_MONTHS)}"
            )
        self.season = season

    @property
    def enabled(self) -> bool:
        """True when an actual season restriction applies (season != 'all')."""
        return self.season != ALL_SEASONS

    def valid_day_indices(self, data_start_year: int, max_days: int) -> list[int]:
        """0-based day-slot indices in ``[0, max_days)`` that fall in the season.

        Returns the full range unchanged when filtering is disabled. Falls back to
        the full range (with a warning) if the season matches no available day, so
        an episode can always start.

        Args:
            data_start_year: Calendar year that day slot 0 (Jan 1) belongs to.
            max_days: Number of selectable day slots available in the data.
        """
        if not self.enabled or max_days <= 0:
            return list(range(max(max_days, 0)))

        months = _SEASON_MONTHS[self.season]
        # Build the whole year's months in one vectorised pass -- avoids a per-day
        # pandas scalar Timedelta add (~13x faster for a full year).
        day_months = pd.date_range(
            pd.Timestamp(year=data_start_year, month=1, day=1),
            periods=max_days,
            freq="D",
        ).month.to_numpy()
        valid = np.nonzero(np.isin(day_months, list(months)))[0].tolist()

        if not valid:
            logger.warning(
                "Season %r matched no day in [0, %d) for year %d; falling back to all days.",
                self.season, max_days, data_start_year,
            )
            return list(range(max_days))
        return valid
