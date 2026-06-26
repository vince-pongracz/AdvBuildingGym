"""``DateProvider`` — capability protocol for the single calendar-date authority.

Lives in ``_common`` (the leaf layer) so ``core`` (``DataVariantManager``) can locate the
provider structurally without importing the concrete ``DateSource`` component, keeping the
dependency direction one-way (``core`` never imports ``components``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DateProvider(Protocol):
    """Structural protocol for the single calendar-date authority (``DateSource``).

    Lets ``DataVariantManager`` resolve the episode's start day without importing the
    concrete ``DateSource`` component — it locates the provider by ``isinstance`` and asks
    for the row count / start year / per-row date.
    """

    def available_rows(self) -> int: ...
    def start_year(self) -> int | None: ...
    def date_at(self, row_offset: int) -> str | None: ...
