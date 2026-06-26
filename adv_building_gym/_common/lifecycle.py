"""Data-reload lifecycle protocols shared across the core ↔ components boundary.

Neutral abstractions components opt into without coupling to a base class. Living in
``_common`` (the leaf layer) keeps the dependency direction one-way: the ``core``
``ReloadDispatcher`` and the ``components`` that implement these contracts both import
*down* into here, so ``core`` never imports ``components``.

The forecasting and calendar-date capability contracts live alongside their consumers:
``Forecastable`` in ``_common.forecasting``, ``DateProvider`` in ``_common.episode_date``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class ReloadObserver(Protocol):
    """Structural protocol: anything with ``on_reload()`` is notified after the host's
    data is reloaded. The host calls it via ``isinstance`` — just implement the method.
    """

    def on_reload(self) -> None: ...


class Reloadable(ABC):
    """A source that accepts a pushed data-variant via ``reload(ds_path)``.

    Pure contract — knows nothing about how the data is stored. 
    ``ReloadDispatcher`` notifies registered ``Reloadable`` sources by name; 
    CSV-free sources simply do not mix this in, they are never reloaded. 
    The CSV-backed implementation (``CsvReloadable``)
    lives in ``components`` because it references the host's composed ``CsvLoader``.
    """
    name: str

    @abstractmethod
    def reload(self, ds_path: str) -> None:
        """Load a new data file for this source."""
