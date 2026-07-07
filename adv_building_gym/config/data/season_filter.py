"""Season-based filtering of selectable episode start days.

Restricts the day-of-year slots a DataCombinator may start an episode on to a
single meteorological season, so train/eval can be scoped to e.g. winter without
touching the day-selection cadence.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from adv_building_gym._common.episode_date import date_column
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

        Returns the full range unchanged when filtering is disabled. Returns an
        **empty list** (never the unfiltered range) if the season matches no
        available day -- e.g. a partial, in-progress calendar year whose data
        hasn't reached the configured season yet. Silently substituting an
        out-of-season day would violate the caller's season restriction, so the
        caller must exclude this (year, data) combination from selection instead
        of consuming an empty result (see ``DataCombinator.get_episode_start_offset``
        and the scenario prefilter in ``config/data/data_config.py``).

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
                "Season %r matched no day in [0, %d) for year %d; this (year, data) "
                "combination cannot serve the season and must be excluded from selection.",
                self.season, max_days, data_start_year,
            )
        return valid

    def contains_season(self, timestamps: pd.Series) -> bool:
        """Whether *timestamps* include at least one real row in the configured season.

        Checks the actual per-row dates directly -- no assumption that row 0 is
        Jan 1 or that rows are evenly spaced/gap-free. Used by ``covers_csv``.
        """
        if not self.enabled:
            return True
        valid = timestamps.dropna()
        if valid.empty:
            return True  # can't determine from unparseable/missing dates -- don't exclude
        months = _SEASON_MONTHS[self.season]
        return bool(valid.dt.month.isin(months).any())

    def covers_csv(self, csv_path: str) -> bool:
        """Whether *csv_path* has at least one real row in the configured season.

        This is the single authority for "does this data belong in the pool" --
        ``DataCombinator._build_variants`` calls it once per scenario (at
        construction time, not per-episode) to drop any (year, data) combination
        whose CSV never reaches the configured season -- e.g. a partial,
        in-progress calendar year -- instead of letting it into the pool where
        an episode could only ever draw an out-of-season day from it.

        Returns True (don't exclude) when filtering is disabled, or when the CSV
        can't be read / has no recognisable date column -- an unreadable file is
        a different problem, not a season mismatch.
        """
        if not self.enabled:
            return True
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except OSError:
            return True
        col = date_column(header)
        if col is None:
            return True
        dates = pd.to_datetime(pd.read_csv(csv_path, usecols=[col])[col], utc=True, errors="coerce")
        return self.contains_season(dates)
