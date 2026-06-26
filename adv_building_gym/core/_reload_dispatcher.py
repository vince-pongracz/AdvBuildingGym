"""ReloadDispatcher — routes data-variant reloads to registered Reloadable sources by name."""

from __future__ import annotations

from typing import Iterable

from adv_building_gym._common.lifecycle import Reloadable


class ReloadDispatcher:
    """Name -> Reloadable registry. Sources register once; 
    ``dispatch`` notifies them per variant.

    Non-Reloadable sources never register, so a variant entry for one is skipped without an
    ``isinstance`` scan or a no-op reload.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, Reloadable] = {}

    def register_all(self, sources: Iterable) -> None:
        """Subscribe every ``Reloadable`` source by its ``name`` (idempotent per name)."""
        for source in sources:
            if isinstance(source, Reloadable):
                self._subscribers[source.name] = source

    def dispatch(self, variant: dict[str, str]) -> None:
        """Reload each registered source named in ``variant`` with its path."""
        for name, path in variant.items():
            source = self._subscribers.get(name)
            if source is not None:
                source.reload(path)
