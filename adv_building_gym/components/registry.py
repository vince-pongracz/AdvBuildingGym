"""Component registry + serialisation for pluggable env components.

Holds ``Serializable`` (the component mixin), ``ComponentRegistry`` (name→class
buckets), and ``from_dict`` (rebuild a registered component from a dict). Lives in
``components/`` so devices/statesources/rewards import it without the config deps.
"""

import inspect
import logging
from abc import ABC
from typing import Any, Dict, Optional, Type, TypeVar, ClassVar, Set

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


T = TypeVar('T', bound='Serializable')


class ComponentRegistry:
    """Registry for mapping class names to their classes for deserialization."""

    _registries: Dict[str, Dict[str, Type['Serializable']]] = {
        'infrastructure': {},
        'statesource': {},
        'reward': {},
    }

    @classmethod
    def register(cls, component_type: str, component_class: Type['Serializable']) -> None:
        """Register a component class under component_type ('infrastructure'/'statesource'/'reward')."""
        if component_type not in cls._registries:
            raise ValueError(f"Unknown component type: {component_type}")
        cls._registries[component_type][component_class.__name__] = component_class
        logger.debug("Registered %s: %s", component_type, component_class.__name__)

    @classmethod
    def get(cls, component_type: str, class_name: str) -> Type['Serializable']:
        """Return the registered class for class_name; ValueError if type/name unknown."""
        if component_type not in cls._registries:
            raise ValueError(f"Unknown component type: {component_type}")
        if class_name not in cls._registries[component_type]:
            available = list(cls._registries[component_type].keys())
            raise ValueError(
                f"Unknown {component_type} class: {class_name}. "
                f"Available: {available}"
            )
        return cls._registries[component_type][class_name]

    @classmethod
    def get_all(cls, component_type: str) -> Dict[str, Type['Serializable']]:
        """Get all registered classes for a component type."""
        return cls._registries.get(component_type, {}).copy()

def _get_serialize_value(param_name: str, value: Any) -> Any:
        """Serialisable form of a param value, or None to skip it."""
        # skip None
        if value is None:
            return None

        # Handle basic types directly
        if isinstance(value, (str, int, float, bool)):
            return value

        # Handle lists of serializable objects
        if isinstance(value, list):
            serialized_list = []
            for item in value:
                if isinstance(item, Serializable):
                    serialized_list.append(item.to_dict())
                elif isinstance(item, (str, int, float, bool)):
                    serialized_list.append(item)
                else:
                    logger.warning(
                        "Skipping non-serializable list item of type %s in %s",
                        type(item).__name__, param_name
                    )
            return serialized_list if serialized_list else None

        # Handle Serializable objects
        if isinstance(value, Serializable):
            return value.to_dict()

        # Skip non-serializable types by default
        logger.debug(
            "Skipping non-serializable param %s of type %s",
            param_name, type(value).__name__
        )
        return None


class Serializable(ABC):
    """JSON serialisation mixin for config components.

    ``_context_params``: param names from context (e.g. building_props), excluded from
    the dict but supplied at reconstruction. ``_exclude_params``: always excluded (e.g.
    internal state). Override ``_get_serialize_value()`` for custom handling.
    """

    # From external context (e.g. building_props); excluded from the dict, supplied at reconstruction
    _context_params: ClassVar[Set[str]] = set()

    # Always excluded (e.g. internal state)
    _exclude_params: ClassVar[Set[str]] = set()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dict via __init__ introspection, excluding context/exclude params."""
        result = {
            'class': self.__class__.__name__,
        }

        # __init__ params = what to serialise
        sig = inspect.signature(self.__class__.__init__)
        params = list(sig.parameters.keys())

        for param in params:
            if param == 'self':
                continue
            if param in self._context_params:
                continue
            if param in self._exclude_params:
                continue

            # Get the value from the instance
            if hasattr(self, param):
                value = getattr(self, param)
                serialized_value = _get_serialize_value(param, value)
                if serialized_value is not None:
                    result[param] = serialized_value

        return result

    @classmethod
    def _get_init_args(
        cls,
        data: Dict[str, Any],
        context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Build __init__ kwargs by merging serialised data with context params."""
        context = context or {}
        kwargs = {}

        # Get __init__ signature
        sig = inspect.signature(cls.__init__)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # from serialised data
            if param_name in data:
                kwargs[param_name] = data[param_name]
            # from context
            elif param_name in context:
                kwargs[param_name] = context[param_name]
            # else fall back to the signature default
            elif param.default is not inspect.Parameter.empty:
                pass
            else:
                # required param missing
                logger.warning(
                    "Missing required parameter '%s' for %s",
                    param_name, cls.__name__
                )

        return kwargs


# ---------------------------------------------------------------------------
# Module-level construction helper (was utils/serialization.py before refactor)
# ---------------------------------------------------------------------------

def from_dict(
    data: Dict[str, Any],
    registry_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> "Serializable":
    """Reconstruct a registered component from a serialised dict.

    Replaces the per-class ``from_dict`` once on StateSource/Infrastructure/RewardFunction.
    """
    class_name = data.get("class")
    if class_name is None:
        raise ValueError(f"Missing 'class' key in {registry_type} data")
    cls = ComponentRegistry.get(registry_type, class_name)
    kwargs = cls._get_init_args(data, context)
    return cls(**kwargs)
