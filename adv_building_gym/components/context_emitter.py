"""ContextEmitter — per-component selection of policy-facing ``ctxt_*`` keys.

Shared base for both ``Infrastructure`` and ``StateSource``. 
Component lists the static-context ``ctxt_*`` keys it wants in the observation space 
via the ``ctxt_keys`` config parameter: a key is published to the obs space **iff** it is named in ``ctxt_keys``
(absent / ``None`` / ``[]`` ⇒ the component emits no ``ctxt_*`` obs key). 
This carries the publish/write helpers plus a fail-fast check for unknown keys.

Use ``ctxt_keys`` on a component's trial-config block in **generalisation** setups where the
underlying parameter varies across episodes and the policy must condition on it. 
Values that only other *components* consume (never the policy) are not routed here — they are written
straight to the ``info`` channel in ``update_state`` (e.g. ``ctxt_building_mC``).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .registry import Serializable


class ContextEmitter(Serializable):
    """Adds the ``ctxt_keys`` allow-list gate to a component's policy-facing ``ctxt_*`` keys."""

    # Class default for components that do not expose ``ctxt_keys`` as a ctor arg
    # (i.e. that publish no ``ctxt_keys``-governed obs key).
    ctxt_keys: list[str] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The ctxt_* keys this component can publish, recorded in ``setup_spaces`` and
        # checked by ``validate_ctxt_keys``. Populated cooperatively via ``super().__init__``.
        self._declared_ctxt_keys: set[str] = set()

    def _publish_ctxt(self, state_spaces: OrderedDict, key: str, box: Any) -> None:
        """Register a policy-facing ``ctxt_*`` space only when named in ``ctxt_keys``.

        Every offered key is recorded (whether or not it is exposed) so
        ``validate_ctxt_keys`` can reject typos.
        """
        self._declared_ctxt_keys.add(key)
        if self.ctxt_keys and key in self.ctxt_keys and key not in state_spaces:
            state_spaces[key] = box

    def _write_ctxt(self, states: dict, key: str, value: Any) -> None:
        """Write a policy-facing ``ctxt_*`` value; a no-op when the key was not exposed."""
        if key in states:
            states[key][0] = value

    def validate_ctxt_keys(self) -> None:
        """Fail fast if ``ctxt_keys`` names a key this component cannot publish (typo guard).

        Must be called after ``setup_spaces`` (which records the offerable keys).
        """
        if not self.ctxt_keys:
            return
        unknown = [key for key in self.ctxt_keys if key not in self._declared_ctxt_keys]
        if unknown:
            name = getattr(self, "name", type(self).__name__)
            raise ValueError(
                f"{type(self).__name__} '{name}': ctxt_keys names unknown context key(s) "
                f"{unknown}; this component can publish {sorted(self._declared_ctxt_keys)}."
            )
