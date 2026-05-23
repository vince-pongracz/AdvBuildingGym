"""Forecastable — interface for StateSources that publish future-step previews.

Separated from ``StateSource`` to follow the Interface Segregation Principle:
only sources that actually need forecasting implement this mixin, so the base
class stays free of CSV-lookahead helpers and ``_forecast_array_cache`` state
that is meaningless for sources without an underlying time series.

Conventions
-----------
* Each forecast key MUST start with ``s_fc_`` and mirror the live ``s_<var>``
  observation in shape and normalisation.
* ``forecast_keys()`` is the static contract used by ``ForecastWrapper`` at
  wrapper-construction time — it must NOT depend on whether a CSV has been
  loaded yet.
* ``forecast()`` returns one list per offset, in the order given by
  ``selected_future_steps``. Implementations should zero-fill (or otherwise
  define a safe default) when data is not yet available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class Forecastable(ABC):
    """Mixin for StateSources that can publish ``s_fc_*`` lookahead values.

    Owns the ``_forecast_array_cache`` field. Subclasses are expected to
    cooperate via ``super().__init__()`` so this constructor runs and the
    cache attribute is initialised before any forecast call.

    Implements the ``ReloadObserver`` structural protocol via ``on_reload``:
    ``StateSource._run_post_load`` calls it after every CSV reload to drop
    cached column views. Hosts must declare the inheritance order
    ``class Foo(StateSource, Forecastable)`` — data class first, mixin second —
    so that ``super().__init__(name=...)`` reaches ``StateSource``, not
    ``Forecastable.__init__`` (which takes no args).
    """

    def __init__(self) -> None:
        super().__init__()
        # Cache of numpy views of CSV columns keyed by column name. Built
        # lazily by ``_csv_forecast`` and cleared on every CSV reload via
        # ``on_reload`` (invoked from ``StateSource._run_post_load``).
        self._forecast_array_cache: dict[str, np.ndarray] = {}

    @abstractmethod
    def forecast_keys(self) -> tuple[str, ...]:
        """Return the ``s_fc_*`` observation keys this source publishes.

        Used by ``ForecastWrapper`` to register entries in the Dict observation
        space at wrapper-construction time — must be data-state independent.
        """

    @abstractmethod
    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        """Return future values for each declared forecast key.

        Args:
            selected_future_steps: Strictly positive integer step offsets,
                relative to ``self.iteration`` (1 = next control step).
                Caller pre-sorts ascending and de-duplicates.

        Returns:
            ``dict[s_fc_<var>, list[float]]`` — values normalised the same
            way as the corresponding live ``s_<var>`` observation key.
            One list entry per offset, in the order given.
        """

    def on_reload(self) -> None:
        """Drop cached column views. Satisfies the ``ReloadObserver`` protocol
        and is called by ``StateSource._run_post_load`` after each CSV reload
        so subsequent ``_csv_forecast`` calls rebuild from the new buffer."""
        self._forecast_array_cache = {}

    def _csv_forecast(
        self,
        ts: Optional[pd.DataFrame],
        effective_index: int,
        column: str,
        selected_future_steps: list[int],
    ) -> list[float]:
        """Read ``ts[column]`` at future indices, zero-fill out-of-range.

        The caller passes ``ts`` and ``effective_index`` explicitly rather than
        having this method reach into the host class — keeps Forecastable
        decoupled from any specific StateSource attribute layout.

        Index per step: ``effective_index + step``. Returns ``0.0`` for any
        index outside ``[0, len(ts))`` or when ``ts is None``.

        Vectorised via numpy advanced indexing — one bulk lookup rather than
        a Python-level loop over ``Series.iloc``.
        """
        if ts is None or column not in ts.columns:
            return [0.0] * len(selected_future_steps)
        arr = self._forecast_array_cache.get(column)
        if arr is None:
            # to_numpy() returns a view for contiguous numeric columns; cached
            # once per CSV so subsequent calls are pure numpy indexing.
            arr = ts[column].to_numpy()
            self._forecast_array_cache[column] = arr
        n = arr.shape[0]
        idxs = np.asarray(selected_future_steps, dtype=np.int64) + effective_index
        mask = (idxs >= 0) & (idxs < n)
        out = np.zeros(idxs.shape[0], dtype=np.float64)
        if mask.any():
            out[mask] = arr[idxs[mask]]
        return out.tolist()
