"""Forecastable — mixin for StateSources that publish future-step previews.

Only forecasting sources implement it, keeping the base class free of CSV-lookahead state.
Conventions:
* Forecast keys start with ``s_fc_`` and mirror the live ``s_<var>`` in shape/normalisation.
* ``forecast_keys()`` is the static contract for ``ForecastWrapper`` — independent of CSV load state.
* ``forecast()`` returns one list per offset (order of ``selected_future_steps``), zero-filling when unavailable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class Forecastable(ABC):
    """Mixin for StateSources publishing ``s_fc_*`` lookahead values.

    Owns ``_forecast_array_cache`` (init via cooperative ``super().__init__()``).
    Satisfies ``ReloadObserver`` via ``on_reload`` (called after each CSV reload to drop
    cached views). Declare hosts as ``class Foo(StateSource, Forecastable)`` (data class first)
    so ``super().__init__(name=...)`` reaches ``StateSource``.
    """

    def __init__(self) -> None:
        super().__init__()
        # numpy views of CSV columns by name; built lazily in _csv_forecast, cleared on reload
        self._forecast_array_cache: dict[str, np.ndarray] = {}

    @abstractmethod
    def forecast_keys(self) -> tuple[str, ...]:
        """The ``s_fc_*`` keys this source publishes (for ``ForecastWrapper``; data-state independent)."""

    @abstractmethod
    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        """Future values per forecast key. ``selected_future_steps``: positive offsets from
        ``self.iteration`` (sorted, deduped). Returns ``dict[s_fc_<var>, list[float]]``,
        normalised like the live ``s_<var>``, one entry per offset.
        """

    def on_reload(self) -> None:
        """Drop cached column views after a CSV reload (ReloadObserver protocol)."""
        self._forecast_array_cache = {}

    def _csv_forecast(
        self,
        ts: Optional[pd.DataFrame],
        effective_index: int,
        column: str,
        selected_future_steps: list[int],
    ) -> list[float]:
        """Read ``ts[column]`` at ``effective_index + step`` for each step, zero-filling
        out-of-range / ``ts is None``. ``ts``/``effective_index`` passed in to stay decoupled
        from the host layout. Vectorised via numpy advanced indexing.
        """
        if ts is None or column not in ts.columns:
            return [0.0] * len(selected_future_steps)
        arr = self._forecast_array_cache.get(column)
        if arr is None:
            # to_numpy() is a view for contiguous numeric columns; cached per CSV
            arr = ts[column].to_numpy()
            self._forecast_array_cache[column] = arr
        n = arr.shape[0]
        idxs = np.asarray(selected_future_steps, dtype=np.int64) + effective_index
        mask = (idxs >= 0) & (idxs < n)
        out = np.zeros(idxs.shape[0], dtype=np.float64)
        if mask.any():
            out[mask] = arr[idxs[mask]]
        return out.tolist()
