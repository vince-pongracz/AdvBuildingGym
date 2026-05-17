"""Lifecycle protocols shared across components.

Defines neutral abstractions that components can opt into without coupling
to any particular base class. Living in ``utils`` keeps the dependency
direction one-way (devices/rewards/envs → utils, never the reverse) and
breaks would-be import cycles between concrete bases and their mixins.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ReloadObserver(Protocol):
    """Structural protocol: anything with ``on_reload()`` is notified after
    a host's underlying data is reloaded.

    The host (typically a ``StateSource``) calls ``self.on_reload()`` when
    ``isinstance(self, ReloadObserver)`` — so mixins simply need to implement
    the method; no registration list, no inheritance from this Protocol.
    """

    def on_reload(self) -> None: ...
