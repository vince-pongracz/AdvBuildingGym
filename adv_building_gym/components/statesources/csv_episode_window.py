"""CsvEpisodeWindow — row-major view of exactly the CSV rows one episode visits."""

from __future__ import annotations

from typing import ClassVar, Optional

import numpy as np

from .csv_column_arrays import CsvColumnArrays


class CsvEpisodeWindow(CsvColumnArrays):
    """Gathers ``_window_columns`` for the episode's rows once per reset.

    ``update_state`` reads every column of ONE row per step, so this layer is row-major
    (``CsvColumnArrays`` stays column-major, which is what the look-ahead gathers need).
    Measured per read: ~0.08 µs from this list of tuples, ~0.6 µs from per-column numpy
    arrays, ~0.85 µs from a numpy 2-D row slice — indexing a list returns a ready-made tuple
    of Python floats, while ``matrix[i]`` allocates an ndarray view and then unboxes each
    scalar. Building the window costs ~0.07 ms per reset.

    Host contract: provides ``iteration`` / ``row_offset``, and calls
    ``build_episode_window`` from its own ``reset()``.
    """

    # Host-provided (StateSource pass-throughs): annotations only, create no attributes.
    iteration: int
    row_offset: int

    # CSV columns gathered per row, in the order update_state unpacks them.
    _window_columns: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        super().__init__()
        self._episode_window: list[tuple[float, ...]] = []

    def _csv_row(self, step: int) -> int:
        """CSV row visited at ``step`` steps into the episode, clamped to the frame.

        Default is the ``row_offset + step`` mapping of ``EnvSync.effective_index``; override
        where the step→row mapping differs (a CSV period other than the control step, or a
        profile that repeats within the episode).
        """
        return min(self.row_offset + step, self.n_rows - 1)

    def _row_values(self, row: int) -> tuple[float, ...]:
        """``_window_columns`` at ``row``; a column the CSV lacks reads 0.0."""
        values = []
        for column in self._window_columns:
            array = self.column_array(column)
            values.append(float(array[row]) if array is not None else 0.0)
        return tuple(values)

    def window_row(self) -> tuple[float, ...]:
        """This step's values for ``_window_columns``.

        Falls back to reading the cached column arrays when no window was built (no
        ``episode_length`` on the info channel, e.g. a component-level test).
        """
        step = self.iteration
        if step < len(self._episode_window):
            return self._episode_window[step]
        return self._row_values(self._csv_row(step))

    def build_episode_window(self, episode_length: Optional[int]) -> None:
        """Gather this episode's rows. Call once per reset, after the row offset is set.

        Sized ``episode_length + 1`` because ``iteration`` reaches ``EPISODE_LENGTH``:
        ``AdvBuildingGym`` increments it before running the exogenous statesources.
        """
        self._episode_window = []
        if not episode_length or not self._window_columns or self.n_rows == 0:
            return

        rows = np.fromiter(
            (self._csv_row(step) for step in range(int(episode_length) + 1)),
            dtype=np.int64,
            count=int(episode_length) + 1,
        )
        columns = [
            array[rows] if (array := self.column_array(column)) is not None
            else np.zeros(rows.shape[0], dtype=np.float64)
            for column in self._window_columns
        ]
        # float32 columns widen to float64 exactly, matching float(np.float32) at the call site.
        self._episode_window = [tuple(row) for row in np.stack(columns, axis=1).tolist()]
