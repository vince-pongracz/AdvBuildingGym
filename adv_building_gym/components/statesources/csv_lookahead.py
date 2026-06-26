"""CsvLookahead — CSV-backed implementation of ``Lookahead`` (cached future-row reads).

Hosts declare ``_lookahead_columns`` (logical channel → CSV column) and mix this in as
``class Foo(StateSource, [Forecastable,] CsvLookahead)``. CSV-free sources implement ``Lookahead``
directly instead.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from .lookahead import Lookahead


class CsvLookahead(Lookahead):
    """Implements ``Lookahead`` by reading ``self.ts`` at ``self.effective_index + step``.

    Host contract: request ``ts`` and ``effective_index`` as pass-through properties. 
    The annotations below declare that dependency without creating attributes.
    """

    # Host-provided (StateSource pass-throughs): the CSV frame and the current row index.
    ts: Optional[pd.DataFrame]
    effective_index: int

    # Logical channel -> CSV column. Hosts override; empty means no lookahead channels.
    _lookahead_columns: ClassVar[dict[str, str]] = {}

    def __init__(self) -> None:
        super().__init__()
        # numpy views of CSV columns, built lazily, cleared on reload
        self._forecast_array_cache: dict[str, np.ndarray] = {}

    def lookahead_keys(self) -> tuple[str, ...]:
        return tuple(self._lookahead_columns.keys())

    def lookahead(self, steps: list[int]) -> dict[str, list[float]]:
        return {
            key: self._csv_forecast(self.ts, self.effective_index, column, steps)
            for key, column in self._lookahead_columns.items()
        }

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
        """Read ``ts[column]`` at ``effective_index + step`` per step; zero-fill OOR / ``ts is None``."""
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
