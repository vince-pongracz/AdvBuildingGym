"""CsvLoader — composition target for time-series-backed StateSources.

CSV-backed sources compose one (``self.loader = CsvLoader(...)``); others hold no
loader and inherit the base's default ``reload`` (which raises).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Project root — three levels up from this file (statesources → devices →
# adv_building_gym → repo root). Mirrors the resolution used previously
# inside StateSource so relative ds_paths keep working from Ray workers
# whose CWD may differ.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Opt-in LRU of parsed CSVs, keyed on the resolved path, bounded by BYTES (the files in play
# span 0.02 MB to 15 MB, so an entry count is the wrong dial). OFF by default.
#
# It is all-or-nothing: measured on the STA trial (20 distinct files, 103 MB working set),
# reset/episode went 110 ms uncached -> 150 ms at a 32 MB budget -> 10 ms at 128 MB. A budget
# that cannot hold the working set is WORSE than no cache, because every miss then pays a
# parse plus an eviction. Since each Ray EnvRunner is a separate process, a budget large
# enough to help costs that much per runner — hence off unless a run opts in with
# ADV_CSV_FRAME_CACHE_MB set above its working set.
#
# The unconditional-reload skip in `reload()` is the default win and needs no memory.
_FRAME_CACHE_BUDGET_MB = float(os.environ.get("ADV_CSV_FRAME_CACHE_MB", "0"))
_frame_cache: "OrderedDict[Path, pd.DataFrame]" = OrderedDict()
_frame_cache_bytes = 0


def _resolve(ds_path: str) -> Path:
    """Absolute path for a ds_path, resolving relative ones against the repo root."""
    resolved = Path(ds_path)
    return resolved if resolved.is_absolute() else _PROJECT_ROOT / resolved


def _read_csv(resolved: Path) -> pd.DataFrame:
    """Parse ``resolved`` (once per process while cached) and return a private copy.

    Hosts mutate their frame in ``_post_load_data_processing`` (adding normalised columns,
    dropping the rest), so each caller must own its copy and the cached frame stays pristine.
    """
    global _frame_cache_bytes

    budget = _FRAME_CACHE_BUDGET_MB * 1e6
    if budget <= 0:
        return pd.read_csv(resolved)

    cached = _frame_cache.get(resolved)
    if cached is not None:
        _frame_cache.move_to_end(resolved)
        return cached.copy()

    cached = pd.read_csv(resolved)
    size = int(cached.memory_usage(deep=True).sum())
    if size <= budget:
        _frame_cache[resolved] = cached
        _frame_cache_bytes += size
        while _frame_cache_bytes > budget:
            _, evicted = _frame_cache.popitem(last=False)
            _frame_cache_bytes -= int(evicted.memory_usage(deep=True).sum())
        # The freshly parsed frame is now the cache's copy; hand the caller its own.
        return cached.copy()
    return cached


class CsvLoader:
    """Owns a time-series CSV and its reload lifecycle.

    Calls the host ``on_reload`` callback after every successful load (incl. the
    initial one), where the host invalidates reload observers and post-processes.
    """

    def __init__(
        self,
        ds_path: Optional[str],
        on_reload: Callable[[], None],
    ) -> None:
        self._on_reload = on_reload
        self.ds_path: Optional[str] = None
        self.ts: Optional[pd.DataFrame] = None
        # Resolved path of the pending read vs. of the frame whose post-processing finished;
        # they differ exactly while on_reload runs for a genuinely new file.
        self._resolved_path: Optional[Path] = None
        self._loaded_path: Optional[Path] = None
        if ds_path is not None:
            self.reload(ds_path)

    @property
    def is_new_data_source(self) -> bool:
        """True when the current ds_path differs from the last processed one.

        Set False after each ``on_reload`` callback completes (which marks the
        path as processed). Hosts read this in their post-processing to gate
        one-time diagnostics such as NaN warnings.
        """
        return self._resolved_path != self._loaded_path

    def reload(self, ds_path: str, force: bool = False) -> None:
        """Load a new CSV, then fire the host's reload callback.

        Re-reading the path already loaded is a no-op: the frame is post-processed and
        nothing mutates it during an episode, so a re-parse would reproduce it exactly.
        The data schedule dispatches a reload every episode regardless of whether the
        variant moved, which made this the dominant cost of ``reset()``. Pass
        ``force=True`` to re-read anyway.
        """
        resolved = _resolve(ds_path)
        self.ds_path = ds_path
        self._resolved_path = resolved

        if not force and self.ts is not None and resolved == self._loaded_path:
            logger.debug("CsvLoader kept the loaded frame for %s", resolved)
            return

        self.ts = _read_csv(resolved)
        self._on_reload()
        self._loaded_path = resolved
        logger.debug("CsvLoader reloaded from %s", resolved)
