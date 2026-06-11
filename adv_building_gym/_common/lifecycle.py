"""Lifecycle protocols shared across components.

Neutral abstractions components opt into without coupling to a base class; living here
keeps the dependency direction one-way and avoids import cycles.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ReloadObserver(Protocol):
    """Structural protocol: anything with ``on_reload()`` is notified after the host's
    data is reloaded. The host calls it via ``isinstance`` — just implement the method.
    """

    def on_reload(self) -> None: ...
