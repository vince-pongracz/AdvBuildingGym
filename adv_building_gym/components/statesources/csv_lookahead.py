"""CsvLookahead — CSV-backed implementation of ``Lookahead`` (cached future-row reads).

Hosts declare ``_lookahead_columns`` (logical channel → CSV column) and mix this in as
``class Foo(StateSource, [Forecastable,] CsvLookahead)``. CSV-free sources implement ``Lookahead``
directly instead.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .csv_column_arrays import CsvColumnArrays
from .lookahead import Lookahead


class CsvLookahead(CsvColumnArrays, Lookahead):
    """Implements ``Lookahead`` by reading ``self.ts`` at ``self.effective_index + step``.

    Column reads and their invalidation come from ``CsvColumnArrays``; this adds the
    future-row gather on top.

    Host contract: request ``ts`` and ``effective_index`` as pass-through properties.
    The annotation below declares that dependency without creating an attribute.
    """

    # Host-provided (StateSource pass-through): the current row index.
    effective_index: int

    # Logical channel -> CSV column. Hosts override; empty means no lookahead channels.
    _lookahead_columns: ClassVar[dict[str, str]] = {}

    def lookahead_keys(self) -> tuple[str, ...]:
        return tuple(self._lookahead_columns.keys())

    def lookahead(self, steps: list[int]) -> dict[str, list[float]]:
        return {
            key: self._csv_forecast(self.effective_index, column, steps)
            for key, column in self._lookahead_columns.items()
        }

    def _csv_forecast(
        self,
        effective_index: int,
        column: str,
        selected_future_steps: list[int],
    ) -> list[float]:
        """Read ``ts[column]`` at ``effective_index + step`` per step; zero-fill OOR / no frame."""
        arr = self.column_array(column)
        if arr is None:
            return [0.0] * len(selected_future_steps)
        n = arr.shape[0]
        idxs = np.asarray(selected_future_steps, dtype=np.int64) + effective_index
        mask = (idxs >= 0) & (idxs < n)
        out = np.zeros(idxs.shape[0], dtype=np.float64)
        if mask.any():
            out[mask] = arr[idxs[mask]]
        return out.tolist()
