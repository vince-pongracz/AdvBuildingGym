"""CsvLoader — composition target for time-series-backed StateSources.

Extracted from ``StateSource`` so the base class no longer assumes every
source loads from a CSV. Sources that need file-backed time series compose
one (``self.loader = CsvLoader(...)``); sources that don't (e.g.
``OperatorEnergyControl``) hold no loader and inherit the base's default
``reload`` which raises ``TypeError``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Project root — three levels up from this file (statesources → devices →
# adv_building_gym → repo root). Mirrors the resolution used previously
# inside StateSource so relative ds_paths keep working from Ray workers
# whose CWD may differ.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class CsvLoader:
    """Owns a single time-series CSV and the lifecycle around reloads.

    The host registers a callback (typically ``StateSource._run_post_load``)
    that is invoked after every successful load — including the initial one
    if a ``ds_path`` was supplied. The callback is where the host fires any
    reload observers (``Forecastable`` cache invalidation) and runs subclass
    post-processing.
    """

    def __init__(
        self,
        ds_path: Optional[str],
        on_reload: Callable[[], None],
    ) -> None:
        self._on_reload = on_reload
        self.ds_path: Optional[str] = None
        self.ts: Optional[pd.DataFrame] = None
        self._last_processed_ds_path: Optional[str] = None
        if ds_path is not None:
            self.reload(ds_path)

    @property
    def is_new_data_source(self) -> bool:
        """True when the current ds_path differs from the last processed one.

        Set False after each ``on_reload`` callback completes (which marks the
        path as processed). Hosts read this in their post-processing to gate
        one-time diagnostics such as NaN warnings.
        """
        return self.ds_path != self._last_processed_ds_path

    def reload(self, ds_path: str) -> None:
        """Load a new CSV, then fire the host's reload callback."""
        resolved = Path(ds_path)
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        self.ds_path = ds_path
        self.ts = pd.read_csv(resolved)
        self._on_reload()
        self._last_processed_ds_path = self.ds_path
        logger.debug("CsvLoader reloaded from %s", resolved)
