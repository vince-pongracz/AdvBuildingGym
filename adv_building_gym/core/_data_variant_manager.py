"""Per-episode data-variant + day-offset selection, keeping reset() free of
variant logic and CSV-row arithmetic.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

from adv_building_gym._common.episode_date import date_column, resolve_episode_date
from adv_building_gym.components.statesources.outer.energy_dyn_price import EnergyPriceDynDataSource
from adv_building_gym.components.statesources.outer.energy_price import EnergyPriceDataSource
from adv_building_gym.components.statesources.outer.weather import WeatherDataSource

if TYPE_CHECKING:
    from adv_building_gym.config.data.data_combinator import DataCombinator
    from adv_building_gym.components.statesources import StateSource

logger = logging.getLogger(__name__)

# Reference time-series for day-offset arithmetic, in priority order. Weather and
# price are full-year 5-min series (the longest sources) and both carry a per-row
# timestamp column, so they alone fix the number of selectable days and the data's
# start year. Anything else is only a defensive fallback.
_REFERENCE_TYPES: tuple[type, ...] = (WeatherDataSource, EnergyPriceDataSource, EnergyPriceDynDataSource)


def _reference_source(
    statesources: Sequence["StateSource"], min_rows: int
) -> Optional["StateSource"]:
    """Pick the source that anchors day arithmetic: the weather source, then the
    price source, else the first remaining series long enough to host an episode."""
    candidates = [s for s in statesources if s.ts is not None and len(s.ts) >= min_rows]
    for ref_type in _REFERENCE_TYPES:
        for src in candidates:
            if isinstance(src, ref_type):
                return src
    return candidates[0] if candidates else None


class DataVariantManager:
    """Tracks episode count, picks a CSV variant, and resolves day offset."""

    def __init__(self, data_combinator: "DataCombinator", episode_length: int) -> None:
        self.data_combinator = data_combinator
        self.episode_length = episode_length
        self.episode_count: int = 0
        self.episode_date: str = ""
        self.episode_day_mode: str = "none"

    def begin_episode(self) -> None:
        self.episode_count += 1

    def select_variant(
        self,
        options: Optional[dict],
        rng: np.random.Generator,
        eval_mode: bool = False,
    ) -> Optional[dict[str, str]]:
        """Select a data variant for this episode.

        Precedence: ``options['data_variant']`` > ``eval_mode`` (fresh random each
        episode, overriding cadence) > the combinator's own cadence (cycle/random).
        """
        if options and "data_variant" in options:
            return options["data_variant"]
        if self.data_combinator.variants:
            return self.data_combinator.get_variant(self.episode_count, rng, force_random=eval_mode)
        return None

    def compute_day_offset(
        self,
        statesources: Sequence["StateSource"],
        rng: np.random.Generator,
        control_step: int,
        options: Optional[dict] = None,
    ) -> int:
        """Compute starting row offset and update episode_date / day_mode.

        """
        steps_per_episode = self.episode_length
        day_offset_source = _reference_source(statesources, steps_per_episode)

        if day_offset_source is not None:
            max_episodes = len(day_offset_source.ts) // steps_per_episode
            date_col = date_column(day_offset_source.ts)
            data_start_year = pd.Timestamp(day_offset_source.ts[date_col].iloc[0]).year if date_col is not None else None
        else:
            max_episodes = 1
            data_start_year = None

        start_row_offset, self.episode_day_mode = self.data_combinator.get_episode_start_offset(
            self.episode_count, max_episodes, steps_per_episode, data_start_year, rng,
        )
        self.episode_date = resolve_episode_date(day_offset_source, start_row_offset, control_step)

        if options and "row_offset" in options:
            start_row_offset = int(options["row_offset"])
            self.episode_date = resolve_episode_date(day_offset_source, start_row_offset, control_step)
            self.episode_day_mode = "manual"

        return start_row_offset
