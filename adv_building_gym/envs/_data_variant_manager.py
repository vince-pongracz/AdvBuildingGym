"""Per-episode data-variant + day-offset selection extracted from AdvBuildingGym.

The env owns *what* runs each episode; this manager owns the choice of
*which slice of data* — keeping reset() free of three-way variant logic
and CSV-row arithmetic.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

from adv_building_gym.utils.episode_date import resolve_episode_date

if TYPE_CHECKING:
    from adv_building_gym.data_combinator import DataCombinator
    from adv_building_gym.devices.statesources import StateSource

logger = logging.getLogger(__name__)


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
    ) -> Optional[dict[str, str]]:
        """Select a data variant for this episode.

        ``options['data_variant']`` (single-env eval / external push)
        wins over the combinator's own choice.
        """
        if options and "data_variant" in options:
            return options["data_variant"]
        if self.data_combinator.variants:
            return self.data_combinator.get_variant(self.episode_count, rng)
        return None

    def compute_day_offset(
        self,
        statesources: Sequence["StateSource"],
        rng: np.random.Generator,
        control_step: int,
        options: Optional[dict] = None,
    ) -> int:
        """Compute starting row offset and update episode_date / day_mode."""
        steps_per_episode = self.episode_length
        max_episodes = 1
        data_start_year = None
        for src in statesources:
            if src.ts is not None and len(src.ts) >= steps_per_episode:
                max_episodes = len(src.ts) // steps_per_episode
                for col in ("start", "start_timestamp", "date", "datetime"):
                    if col in src.ts.columns:
                        data_start_year = pd.Timestamp(src.ts[col].iloc[0]).year
                        break
                break

        start_row_offset, self.episode_day_mode = self.data_combinator.get_episode_start_offset(
            self.episode_count, max_episodes, steps_per_episode, data_start_year, rng,
        )
        self.episode_date = resolve_episode_date(statesources, start_row_offset, control_step)

        if options and "row_offset" in options:
            start_row_offset = int(options["row_offset"])
            self.episode_date = resolve_episode_date(statesources, start_row_offset, control_step)
            self.episode_day_mode = "manual"

        return start_row_offset
