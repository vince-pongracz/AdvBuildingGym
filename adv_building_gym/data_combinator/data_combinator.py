"""Schedules CSV data source variants across training episodes."""

import itertools
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataCombinator:
    """Schedules CSV data source variants across training episodes.

    Separates correlated sources (scenarios) from independent ones (variable):

    Args:
        scenarios: Explicit variant bundles for correlated sources (e.g. weather + price).
                   Each entry is a dict[source_name, path]. These are never cross-producted
                   with each other -- they advance as a unit.
        variable:  Maps source_name -> list[paths] for sources independent of everything else
                   (e.g. EV profiles). Cartesian product is applied across variable axes.
        swap_every_n_episodes: Advance to the next variant every N episodes.
        mode: "cycle" (round-robin) or "random".
        day: Controls which day of the CSV data each episode starts at.
             - ``"random"`` (default): sample a uniformly random day each episode.
             - ``"each"``: walk through days sequentially, advancing every episode.
             - A date string (e.g. ``"2025-03-15"``): pin every episode to that
               specific calendar day.

    Final pool = scenarios x variable_combinations.
    If scenarios is empty, only variable combinations are used (and vice versa).
    """

    scenarios: list[dict[str, str]] = field(default_factory=list)
    variable: dict[str, list[str]] = field(default_factory=dict)
    swap_every_n_episodes: int = 5
    mode: Literal["cycle", "random"] = "cycle"
    day: str = "random"
    seed: int = 42
    shuffle: bool = True
    _day_date = None  # Cached parsed date for day mode
    _variants: list[dict[str, str]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._variants = self._build_variants()

    @property
    def variants(self) -> list[dict[str, str]]:
        """Return the cached variant pool."""
        return self._variants

    def _build_variants(self) -> list[dict[str, str]]:
        """Build the full variant pool from scenarios x variable combinations."""
        # Build variable combinations (Cartesian product of independent axes)
        if self.variable:
            keys = list(self.variable.keys())
            variable_combos: list[dict[str, str]] = [
                dict(zip(keys, combo))
                # NOTE: itertools.product(*[self.variable[k] for k in keys]) gives Cartesian product of the variable lists.
                # We want to keep track of which path belongs to which source_name, so we zip back with keys to get dicts.
                for combo in itertools.product(*(self.variable[k] for k in keys))
            ]
        else:
            variable_combos = [{}]  # neutral element -- no independent sources

        # Cross-product: each scenario x each variable combination
        if self.scenarios:
            pool = [
                {**scenario, **var_combo}
                for scenario in self.scenarios
                for var_combo in variable_combos
            ]
        else:
            # No scenarios -- return variable combinations only (omit the empty-dict case)
            pool = variable_combos if self.variable else []

        if self.shuffle:
            rng = np.random.default_rng(seed=self.seed)
            rng.shuffle(pool)

        logger.info(
            "Generated %d variant(s) (%d scenario(s) x %d variable combo(s), swap every %d episode(s), mode=%s, day=%s)",
            len(pool), len(self.scenarios), len(variable_combos),
            self.swap_every_n_episodes, self.mode, self.day,
        )
        return pool

    def get_variant(
        self, episode_count: int, rng: np.random.Generator | None = None
    ) -> dict[str, str]:
        """Return the variant dict for the given episode count.

        Args:
            episode_count: Current episode number (used for cycling / random selection).
            rng: Optional NumPy Generator for random mode.

        Returns:
            A dict mapping source_name -> file path for sources that should be reloaded.
            Empty dict if no variants are configured.
        """
        pool = self.variants
        if not pool:
            return {}
        idx = self._variant_pool_index(episode_count, rng)
        return pool[idx]

    def _variant_pool_index(
        self, episode_count: int, rng: np.random.Generator | None = None
    ) -> int:
        """Return the pool index for the given episode count."""
        pool = self.variants
        if self.mode == "random" and rng is not None:
            return int(rng.integers(0, len(pool)))
        return (episode_count // self.swap_every_n_episodes) % len(pool)

    def get_day_offset(
        self,
        episode_count: int,
        max_days: int,
        steps_per_day: int,
        data_start_year: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, str]:
        """Compute the row offset for the day selection of the current episode.

        Day selection happens every episode, independent of variant swapping:

        - ``"random"``: sample a uniformly random day each episode.
        - ``"each"``: walk through days sequentially (episode 1 → day 0,
          episode 2 → day 1, …), wrapping around when all days are exhausted.
        - A date string (e.g. ``"2025-03-15"``): pin every episode to that day.

        Args:
            episode_count: Current episode number.
            max_days: Number of complete days available in the data.
            steps_per_day: Rows per day (e.g. 288 for 5-min steps).
            data_start_year: Year the data begins (derived from the CSV).
                Falls back to the current year if None.
            rng: NumPy Generator for random day selection.

        Returns:
            (row_offset, day_mode) where row_offset is the starting row and
            day_mode is the ``day`` value for logging.
        """
        if self.day == "random":
            if rng is not None:
                day_index = int(rng.integers(0, max_days))
            else:
                day_index = 0
        elif self.day == "each":
            day_index = episode_count % max_days
        else:
            # Interpret as a date string — find the matching day index
            day_index = self._date_to_day_index(self.day)

        year = data_start_year if data_start_year is not None else pd.Timestamp.now().year
        jan1 = pd.Timestamp(year=year, month=1, day=1)
        self._day_date = jan1 + pd.Timedelta(days=day_index)

        row_offset = day_index * steps_per_day
        return row_offset, self.day

    def get_day_date(self) -> str | None:
        """Return the date string of the most recently selected day, or None."""
        if self._day_date is not None:
            return self._day_date.strftime("%Y-%m-%d")
        return None

    @staticmethod
    def _date_to_day_index(date_str: str) -> int:
        """Convert a date string like '2025-03-15' to a day-of-year index (0-based)."""
        date = pd.Timestamp(date_str)
        return date.day_of_year - 1

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "swap_every_n_episodes": self.swap_every_n_episodes,
            "mode": self.mode,
            "day": self.day,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "scenarios": self.scenarios,
            "variable": self.variable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataCombinator":
        """Reconstruct a DataCombinator from a dictionary."""
        return cls(
            swap_every_n_episodes=d.get("swap_every_n_episodes", 5),
            mode=d.get("mode", "cycle"),
            day=d.get("day", "random"),
            seed=d.get("seed", 42),
            shuffle=d.get("shuffle", True),
            scenarios=d.get("scenarios", []),
            variable=d.get("variable", {}),
        )
