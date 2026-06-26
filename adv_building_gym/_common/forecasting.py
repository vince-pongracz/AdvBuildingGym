"""``Forecastable`` — capability interface for ``s_fc_*`` look-ahead observations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Forecastable(ABC):
    """Interface for components publishing ``s_fc_*`` or ``ctxt_fc_*`` look-ahead observations.

    Contract only — no state, no machinery.
    Implement on statesources or infrastructures that contribute forecast keys;
    ``ForecastWrapper`` discovers them via ``isinstance``.

    Conventions:
    * Forecast keys mirror live observation key with an ``fc_`` segment inserted after the
      frame prefix:
      - ``s_fc_<var>`` ↔ ``s_<var>`` (normalised states)
      - ``ctxt_fc_<var>`` ↔ ``ctxt_<var>`` (slow-changing context scalars), same shape/normalisation.
    * ``forecast_keys()`` is the static contract for ``ForecastWrapper``
    * ``forecast()`` returns one list per offset (order of ``selected_future_steps``), zero-filling when unavailable.
    """

    @abstractmethod
    def forecast_keys(self) -> tuple[str, ...]:
        """The ``fc_*`` keys this component publishes (for ``ForecastWrapper``; data-state independent)."""

    @abstractmethod
    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        """Future values per forecast key. ``selected_future_steps``: positive offsets from
        ``self.iteration`` (sorted, deduped). Returns ``dict[s_fc_<var>, list[float]]``,
        normalised like the live ``s_<var>``, one entry per offset.
        """
