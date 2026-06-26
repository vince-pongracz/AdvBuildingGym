"""Per-episode data-variant + day-offset selection, keeping reset() free of
variant logic and CSV-row arithmetic.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from adv_building_gym._common.date_provider import DateProvider
from adv_building_gym._common.episode_date import resolve_episode_date

if TYPE_CHECKING:
    from adv_building_gym.config.data.data_combinator import DataCombinator
    from adv_building_gym.components.statesources import StateSource

logger = logging.getLogger(__name__)


def _date_source(statesources: Sequence["StateSource"]) -> Optional[DateProvider]:
    """The single date authority — sole owner of day-offset arithmetic and the episode date.

    Located structurally (``DateProvider`` protocol) so core needn't import the concrete
    ``DateSource`` component.
    """
    return next((s for s in statesources if isinstance(s, DateProvider)), None)


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
        date_source = _date_source(statesources)

        if date_source is not None and date_source.available_rows() >= steps_per_episode:
            max_episodes = date_source.available_rows() // steps_per_episode
            data_start_year = date_source.start_year()
        else:
            max_episodes = 1
            data_start_year = datetime.now().year

        start_row_offset, self.episode_day_mode = self.data_combinator.get_episode_start_offset(
            self.episode_count, max_episodes, steps_per_episode, data_start_year, rng,
        )
        self.episode_date = resolve_episode_date(date_source, start_row_offset, control_step)

        if options and "row_offset" in options:
            start_row_offset = int(options["row_offset"])
            self.episode_date = resolve_episode_date(date_source, start_row_offset, control_step)
            self.episode_day_mode = "manual"

        return start_row_offset
