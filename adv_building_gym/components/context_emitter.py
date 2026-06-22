"""ContextEmitter — per-component switch for publishing policy-facing ``ctxt_*`` keys.

Shared base for both ``Infrastructure`` and ``StateSource``. 
The flag is an ``__init__`` config parameter on each gateable component.
This base only carries the two gating helpers and a safe default.

Use ``emit_ctxt: true`` on a component's trial-config block in **generalisation** setups
where the underlying parameter varies across episodes and the policy must condition on it.
Leave it off (default) for single-config runs, where a constant ``ctxt_*`` only biases the
policy. Only *policy-only* ctxt (read by no other component/reward) are routed through the
helpers; *functional* ctxt (data channels other components read) are published directly.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .registry import Serializable


class ContextEmitter(Serializable):
    """Adds the ``emit_ctxt`` gate to a component's policy-only ``ctxt_*`` keys."""

    # Class default for components that do not expose ``emit_ctxt`` as a ctor arg.
    emit_ctxt: bool = False

    def _publish_ctxt(self, state_spaces: OrderedDict, key: str, box: Any) -> None:
        """Register a policy-only ``ctxt_*`` space only when this component opts in."""
        if self.emit_ctxt and key not in state_spaces:
            state_spaces[key] = box

    def _write_ctxt(self, states: dict, key: str, value: Any) -> None:
        """Write a policy-only ``ctxt_*`` value; a no-op when the key was gated off."""
        if key in states:
            states[key][0] = value
