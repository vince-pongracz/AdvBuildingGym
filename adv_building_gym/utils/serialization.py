"""Serialization helpers for registry-backed components.

Module-level functions that replace the three byte-identical ``from_dict``
classmethods that previously lived on ``StateSource``, ``Infrastructure``,
and ``RewardFunction``. Centralising the registry lookup + kwargs assembly
removes a long-running source of duplication and keeps each base class
focused on its runtime responsibilities.

Components still own their own ``to_dict`` and ``_get_serialize_value`` via
the ``Serializable`` mixin — only construction-from-dict was duplicated and
is now consolidated here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .serializable import ComponentRegistry, Serializable


def from_dict(
    data: Dict[str, Any],
    registry_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Serializable:
    """Reconstruct a registered component from a serialised dict.

    Args:
        data: Dictionary with a ``"class"`` key naming the concrete component
            class and the remaining ctor parameters.
        registry_type: Which ``ComponentRegistry`` bucket to look up
            (``"statesource"`` / ``"infrastructure"`` / ``"reward"``).
        context: Optional extra kwargs supplied at reconstruction time
            (``building_props``, ``control_step``, …) that aren't serialised
            into ``data``.

    Returns:
        Reconstructed component instance.

    Raises:
        ValueError: If ``data`` has no ``"class"`` key or the class is not
            registered under ``registry_type``.
    """
    class_name = data.get("class")
    if class_name is None:
        raise ValueError(f"Missing 'class' key in {registry_type} data")
    cls = ComponentRegistry.get(registry_type, class_name)
    kwargs = cls._get_init_args(data, context)
    return cls(**kwargs)
