"""CSV-backed ``Reloadable`` implementation, detected via ``isinstance``.

The pure ``Reloadable`` contract lives in ``_common.lifecycle`` (so ``core`` can import it
without importing ``components``); this module adds the CSV-specific implementation that
delegates to the host's composed ``CsvLoader``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from adv_building_gym._common.lifecycle import Reloadable

if TYPE_CHECKING:
    from .csv_loader import CsvLoader

logger = logging.getLogger(__name__)

__all__ = ["Reloadable", "CsvReloadable"]


class CsvReloadable(Reloadable):
    """``Reloadable`` implemented by delegating to the host's composed ``CsvLoader``.

    Host contract: provides ``name`` and sets ``loader``.
    These are declared below so the mix-in's collaborators are explicit and
    type-checkable instead of being reached for as undeclared attributes; the
    guard in ``reload`` turns a missing loader into a clear error.
    """

    # Supplied by the host StateSource (annotations only — no class attributes created).
    name: str
    loader: Optional[CsvLoader]

    def reload(self, ds_path: str) -> None:
        if self.loader is None:
            raise RuntimeError(
                f"{type(self).__name__} '{getattr(self, 'name', '<unnamed>')}' mixes in "
                "CsvReloadable but never set self.loader = CsvLoader(...); cannot reload."
            )
        self.loader.reload(ds_path)
        logger.debug("Reloaded '%s' from %s", self.name, ds_path)
