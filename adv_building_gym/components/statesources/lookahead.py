"""Lookahead — backing-agnostic interface for future raw values of named channels.

Unlike ``Forecastable`` (which publishes normalised ``s_fc_*`` into the obs space), ``Lookahead``
only provides raw values to other components. Backed by CSV rows (``CsvLookahead``) or a formula.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Lookahead(ABC):
    """Provides future raw values of named channels, relative to the current ``effective_index``."""

    @abstractmethod
    def lookahead_keys(self) -> tuple[str, ...]:
        """Logical channels, e.g. ``('baseprice',)`` or ``('solar_W_m2', 'wind_ms')``."""

    @abstractmethod
    def lookahead(self, steps: list[int]) -> dict[str, list[float]]:
        """Raw values at ``effective_index + step`` per channel; zero-filled out of range."""
