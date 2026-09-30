"""CsvColumnArrays — cached 1-D numpy views of the host's CSV columns.

Mixed in by every source that reads ``self.ts`` at step frequency, either directly
(``update_state``) or through ``CsvLookahead``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class CsvColumnArrays:
    """Serves ``self.ts`` columns as cached 1-D numpy arrays.

    Going through pandas per step is expensive: ``ts["col"].iloc[i]`` costs ~7 µs and
    ``ts.iloc[i]`` ~43 µs (it rebuilds a Series, upcasting when column dtypes differ),
    against ~0.4 µs off a cached array. A frame is only ever replaced or mutated during a
    reload, so a view stays valid for the life of the loaded CSV.

    Host contract: provides ``ts``, and routes ``on_reload`` through
    ``StateSource._run_post_load`` (the ``ReloadObserver`` protocol).
    """

    # Host-provided (StateSource pass-through): annotation only, creates no attribute.
    ts: Optional[pd.DataFrame]

    def __init__(self) -> None:
        super().__init__()
        # Lazily built per column, dropped on reload.
        self._column_arrays: dict[str, np.ndarray] = {}
        self._n_rows: int = 0

    @property
    def n_rows(self) -> int:
        """Row count of the loaded frame; 0 without one.

        Cached because clamping every step read costs a ``len(self.ts)`` call (~0.4 µs)
        that only changes on reload.
        """
        if self._n_rows == 0 and self.ts is not None:
            self._n_rows = len(self.ts)
        return self._n_rows

    def column_array(self, column: str) -> Optional[np.ndarray]:
        """Cached 1-D view of ``ts[column]``; ``None`` when there is no frame or column.

        Absent columns are not cached, so a column added later by
        ``_post_load_data_processing`` is picked up on the next call.
        """
        array = self._column_arrays.get(column)
        if array is not None:
            return array

        frame = self.ts
        if frame is None or column not in frame.columns:
            return None
        # to_numpy() is a view for contiguous numeric columns; cached per CSV.
        array = frame[column].to_numpy()
        self._column_arrays[column] = array
        return array

    def on_reload(self) -> None:
        """Drop cached views after a CSV reload (ReloadObserver protocol).

        Runs before ``_post_load_data_processing``, so any view rebuilt afterwards
        reflects the post-processed frame.
        """
        self._column_arrays = {}
        self._n_rows = 0
