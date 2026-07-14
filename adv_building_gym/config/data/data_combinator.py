"""Schedules CSV data source variants across training episodes."""

import itertools
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from adv_building_gym.config.data.season_filter import SeasonFilter

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
        season: Restricts ``random``/``each`` day selection to one meteorological
                season (``winter``/``spring``/``summer``/``autumn``/
                ``spring_autumn``); ``"all"`` (default) disables filtering. A
                pinned date string ignores this. See ``season_filter.py``.
        active_source_names: Names of the trial's reloadable statesources. When set,
                variant keys with no consuming statesource are excluded: variable axes
                not in the set are disabled, scenario dicts are projected onto the set,
                and the resulting duplicates are collapsed — a CSV only constitutes a
                variant if a configured statesource processes it. ``None`` disables
                filtering (full pool, legacy behaviour).

    Final pool = scenarios x variable_combinations.
    If scenarios is empty, only variable combinations are used (and vice versa).
    """

    scenarios: list[dict[str, str]] = field(default_factory=list)
    variable: dict[str, list[str]] = field(default_factory=dict)
    active_source_names: frozenset[str] | None = None

    swap_every_n_episodes: int = 5
    mode: Literal["cycle", "random"] = "cycle"
    day: str = "random"
    season: str = "all"

    seed: int = 42
    shuffle: bool = True
    _day_date = None  # Cached parsed date for day mode
    _variants: list[dict[str, str]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._season_filter = SeasonFilter(self.season)
        self._variants = self._build_variants()

    @property
    def variants(self) -> list[dict[str, str]]:
        """Return the cached variant pool."""
        return self._variants

    def _build_variants(self) -> list[dict[str, str]]:
        """Build the full variant pool from scenarios x variable combinations."""
        scenarios = self._season_compatible_scenarios()
        variable = self.variable
        if self.active_source_names is not None:
            # Season filtering above must see the unprojected dicts (it reads the
            # "date"/"weather" paths); conformance filtering comes after.
            scenarios = self._project_scenarios_onto_active_sources(scenarios)
            variable = self._active_variable_axes()

        # Build variable combinations (Cartesian product of independent axes)
        if variable:
            keys = list(variable.keys())
            variable_combos: list[dict[str, str]] = [
                dict(zip(keys, combo))
                # NOTE: itertools.product(*[variable[k] for k in keys]) gives Cartesian product of the variable lists.
                # We want to keep track of which path belongs to which source_name, so we zip back with keys to get dicts.
                for combo in itertools.product(*(variable[k] for k in keys))
            ]
        else:
            variable_combos = [{}]  # neutral element -- no independent sources

        # Cross-product: each scenario x each variable combination
        if scenarios:
            pool = [
                {**scenario, **var_combo}
                for scenario in scenarios
                for var_combo in variable_combos
            ]
        else:
            # No scenarios -- return variable combinations only (omit the empty-dict case)
            pool = variable_combos if variable else []

        if not pool and (self.scenarios or self.variable):
            logger.warning(
                "Data variant pool is empty after filtering: no configured statesource "
                "consumes any of the scheduled data sources.",
            )

        if self.shuffle:
            rng = np.random.default_rng(seed=self.seed)
            rng.shuffle(pool)

        logger.info(
            "Generated %d variant(s) (%d scenario(s) x %d variable combo(s), swap every %d episode(s), mode=%s, day=%s, season=%s)",
            len(pool), len(scenarios), len(variable_combos),
            self.swap_every_n_episodes, self.mode, self.day, self.season,
        )
        return pool

    def _season_compatible_scenarios(self) -> list[dict[str, str]]:
        """Drop scenarios whose data never reaches the configured season.

        Checked once here (construction time), not per-episode: the only question
        is "does this scenario's CSV cover the season" (``SeasonFilter.covers_csv``),
        resolved from the "date" source (falling back to "weather"). A scenario that
        fails this -- e.g. a partial, in-progress calendar year -- is excluded from
        the pool entirely and logged, rather than being offered as a candidate that
        could only ever yield an out-of-season day.
        """
        if not self._season_filter.enabled or not self.scenarios:
            return self.scenarios

        coverage_cache: dict[str, bool] = {}
        kept: list[dict[str, str]] = []
        for scenario in self.scenarios:
            csv_path = scenario.get("date", scenario.get("weather"))
            if csv_path is None:
                kept.append(scenario)  # nothing to check against -- keep it
                continue
            if csv_path not in coverage_cache:
                coverage_cache[csv_path] = self._season_filter.covers_csv(csv_path)
            if coverage_cache[csv_path]:
                kept.append(scenario)
            else:
                logger.warning(
                    "Excluding scenario from data schedule: season %r has no day in %s",
                    self.season, csv_path,
                )
        return kept

    def _active_variable_axes(self) -> dict[str, list[str]]:
        """Drop variable axes with no consuming statesource in the trial config.

        A CSV only constitutes a variant if a configured reloadable statesource
        processes it — a dead axis would multiply the pool with duplicates.
        """
        disabled = [axis for axis in self.variable if axis not in self.active_source_names]
        for axis in disabled:
            logger.warning(
                "Variable axis %r disabled: no reloadable statesource with that name "
                "in the trial config.", axis,
            )
        return {axis: paths for axis, paths in self.variable.items() if axis not in disabled}

    def _project_scenarios_onto_active_sources(self, 
        scenarios: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Project scenario dicts onto the active sources and collapse duplicates.

        Keys with no consuming statesource (e.g. an ``E_price`` CSV while the trial
        uses the synthetic fix-price source) are removed; scenarios that become
        identical after projection are deduplicated order-preservingly. Scenarios
        projected to the empty dict are dropped entirely.
        """
        dead_keys = {key for scenario in scenarios for key in scenario if key not in self.active_source_names}
        if dead_keys:
            logger.warning(
                "Scenario source(s) %s disabled: no reloadable statesource with that "
                "name in the trial config.", sorted(dead_keys),
            )
        projected: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for scenario in scenarios:
            kept = {name: path for name, path in scenario.items() if name in self.active_source_names}
            if not kept:
                continue
            dedupe_key = tuple(sorted(kept.items()))
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                projected.append(kept)
        if len(projected) < len(scenarios):
            logger.info(
                "Scenario pool reduced %d -> %d after projecting onto the trial's "
                "statesources and deduplicating.", len(scenarios), len(projected),
            )
        return projected

    def get_variant(
        self,
        episode_count: int,
        rng: np.random.Generator | None = None,
        force_random: bool = False,
    ) -> dict[str, str]:
        """Return the variant dict for the given episode count.

        Args:
            episode_count: Current episode number (used for cycling / random selection).
            rng: Optional NumPy Generator for random mode.
            force_random: When True, draw a uniformly random variant regardless
                of ``mode`` (used by evaluation runners so each eval episode
                samples an independent variant instead of following the
                ``episode_count // swap_every_n_episodes`` cadence).

        Returns:
            A dict mapping source_name -> file path for sources that should be reloaded.
            Empty dict if no variants are configured.
        """
        pool = self.variants
        if not pool:
            return {}
        idx = self._get_variant_pool_index(episode_count, rng, force_random=force_random)
        return pool[idx]

    def _get_variant_pool_index(
        self,
        episode_count: int,
        rng: np.random.Generator | None = None,
        force_random: bool = False,
    ) -> int:
        """Return the pool index for the given episode count."""
        variants_pool = self.variants
        if (force_random or self.mode == "random") and rng is not None:
            return int(rng.integers(0, len(variants_pool)))
        # Cycle through variants every swap_every_n_episodes episodes
        n_variant_on_the_schedule = episode_count // self.swap_every_n_episodes
        return n_variant_on_the_schedule % len(variants_pool)

    def get_episode_start_offset(
        self,
        episode_count: int,
        max_days: int,
        steps_per_day: int,
        data_start_year: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, str]:
        """Row offset for this episode's start day (independent of variant swapping).

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
        year = data_start_year if data_start_year is not None else pd.Timestamp.now().year

        if self.day == "random":
            # Restrict the draw to in-season days only (full range when season="all").
            # _season_compatible_scenarios() already keeps only scenarios that cover
            # the season, so valid_days is never empty for a real, selected variant.
            valid_days = self._season_filter.valid_day_indices(year, max_days)
            if rng is not None:
                day_index = int(rng.choice(valid_days))
            else:
                day_index = valid_days[0]
        elif self.day == "each":
            # Walk sequentially through in-season days only.
            valid_days = self._season_filter.valid_day_indices(year, max_days)
            day_index = valid_days[episode_count % len(valid_days)]
        else:
            # Interpret as a date string — find the matching day index (season ignored:
            # an explicitly pinned calendar day always wins).
            day_index = self._date_to_day_index(self.day)

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
