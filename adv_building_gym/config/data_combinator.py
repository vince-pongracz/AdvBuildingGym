"""Schedules CSV data source variants across training episodes."""

import itertools
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

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

    Final pool = scenarios x variable_combinations.
    If scenarios is empty, only variable combinations are used (and vice versa).
    """

    scenarios: list[dict[str, str]] = field(default_factory=list)
    variable: dict[str, list[str]] = field(default_factory=dict)
    swap_every_n_episodes: int = 1
    mode: Literal["cycle", "random"] = "cycle"

    @property
    def variants(self) -> list[dict[str, str]]:
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
            return [
                # ** operator unpacks a dictionary into keyword arguments -- we merge the scenario dict and variable combo dict
                {**scenario, **var_combo}
                for scenario in self.scenarios
                for var_combo in variable_combos
            ]
        # No scenarios -- return variable combinations only (omit the empty-dict case)
        return variable_combos if self.variable else []

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
        if self.mode == "random" and rng is not None:
            return pool[int(rng.integers(0, len(pool)))]
        return pool[(episode_count // self.swap_every_n_episodes) % len(pool)]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "swap_every_n_episodes": self.swap_every_n_episodes,
            "mode": self.mode,
            "scenarios": self.scenarios,
            "variable": self.variable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataCombinator":
        """Reconstruct a DataCombinator from a dictionary."""
        return cls(
            swap_every_n_episodes=d.get("swap_every_n_episodes", 1),
            mode=d.get("mode", "cycle"),
            scenarios=d.get("scenarios", []),
            variable=d.get("variable", {}),
        )
