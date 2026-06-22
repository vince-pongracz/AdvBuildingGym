"""CsvLookahead mixin: cached future-row CSV reads, kept separate from ``Forecastable``.

A component may need lookahead reads WITHOUT publishing forecasts itself — e.g.
``WeatherDataSource`` supplies future irradiance/wind to the generators' power forecasts
but exposes no ``s_fc_*`` of its own. Such a source uses ``CsvLookahead`` and is *not*
``Forecastable`` (see ``forecastable.py`` for the publishing interface).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class CsvLookahead:
    """Mixin: cached future-row CSV reads (``_csv_forecast``), independent of ``Forecastable``.

    Owns ``_forecast_array_cache`` (init via cooperative ``super().__init__()``). Satisfies
    ``ReloadObserver`` via ``on_reload`` (called after each CSV reload to drop cached views).
    Declare hosts as ``class Foo(StateSource, [Forecastable,] CsvLookahead)`` (data class first)
    so ``super().__init__(name=...)`` reaches ``StateSource``.
    """

    def __init__(self) -> None:
        super().__init__()
        # numpy views of CSV columns by name; built lazily in _csv_forecast, cleared on reload
        self._forecast_array_cache: dict[str, np.ndarray] = {}

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
