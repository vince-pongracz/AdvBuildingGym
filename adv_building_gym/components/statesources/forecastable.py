"""Forecasting interface for components publishing ``s_fc_*`` look-ahead observations.

``Forecastable`` is a *pure interface*: implement it only when a component actually
publishes ``s_fc_*`` look-ahead observations. ``ForecastWrapper`` queries every
Forecastable in the env (statesources and infrastructure). Components needing only
cached future-row CSV reads — WITHOUT publishing forecasts — use ``CsvLookahead``
(in ``csv_lookahead.py``) instead and are *not* ``Forecastable``.

Conventions:
* Forecast keys start with ``s_fc_`` and mirror the live ``s_<var>`` in shape/normalisation.
* ``forecast_keys()`` is the static contract for ``ForecastWrapper`` — independent of CSV load state.
* ``forecast()`` returns one list per offset (order of ``selected_future_steps``), zero-filling when unavailable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Forecastable(ABC):
    """Interface for components publishing ``s_fc_*`` look-ahead observations.

    Pure contract — no state, no machinery. Implement on statesources or infrastructure
    that contribute forecast keys; ``ForecastWrapper`` discovers them via ``isinstance``.
    Sources needing only cached CSV lookahead (no published forecast) use ``CsvLookahead``
    instead, so they are not forced to stub these methods.
    """

    @abstractmethod
    def forecast_keys(self) -> tuple[str, ...]:
        """The ``s_fc_*`` keys this component publishes (for ``ForecastWrapper``; data-state independent)."""

    @abstractmethod
    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        """Future values per forecast key. ``selected_future_steps``: positive offsets from
        ``self.iteration`` (sorted, deduped). Returns ``dict[s_fc_<var>, list[float]]``,
        normalised like the live ``s_<var>``, one entry per offset.
        """
