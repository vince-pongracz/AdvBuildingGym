"""Serialization utilities for config components.

This module provides a flexible serialization system for Infrastructure, StateSource,
and RewardFunction components that doesn't hard-code specific attributes.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar, ClassVar, Set

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
        """
        Register a component class.

        Args:
            component_type: Type of component ('infrastructure', 'statesource', 'reward')
            component_class: The class to register
        """
        if component_type not in cls._registries:
            raise ValueError(f"Unknown component type: {component_type}")
        cls._registries[component_type][component_class.__name__] = component_class
        logger.debug("Registered %s: %s", component_type, component_class.__name__)

    @classmethod
    def get(cls, component_type: str, class_name: str) -> Type['Serializable']:
        """
        Get a component class by name.

        Args:
            component_type: Type of component ('infrastructure', 'statesource', 'reward')
            class_name: Name of the class to retrieve

        Returns:
            The registered class

        Raises:
            ValueError: If component type or class name is not found
        """
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


class Serializable(ABC):
    """
    Mixin providing flexible JSON serialization for config components.

    Subclasses can customize serialization by:
    - Setting `_context_params`: Set of parameter names that come from context
      (e.g., building_props) and should be excluded from serialization
    - Setting `_exclude_params`: Set of parameter names to always exclude
    - Overriding `_get_serialize_value()` for custom value handling

    Context parameters are not stored in the serialized dict but are passed
    during reconstruction from the deserialization context.
    """

    # Parameters that come from external context (e.g., building_props, infras)
    # These are excluded from serialization but required during reconstruction
    _context_params: ClassVar[Set[str]] = set()

    # Parameters to always exclude from serialization (e.g., internal state)
    _exclude_params: ClassVar[Set[str]] = set()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        Uses introspection of __init__ parameters to determine what to serialize.
        Excludes context_params and exclude_params.

        Returns:
            Dictionary representation of this component
        """
        result = {
            'class': self.__class__.__name__,
        }

        # Get __init__ signature to find serializable parameters
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
                serialized_value = self._get_serialize_value(param, value)
                if serialized_value is not None:
                    result[param] = serialized_value

        return result

    def _get_serialize_value(self, param_name: str, value: Any) -> Any:
        """
        Get the serializable value for a parameter.

        Override this method to handle special serialization cases.

        Args:
            param_name: Name of the parameter
            value: Current value of the parameter

        Returns:
            Serializable value, or None to skip this parameter
        """
        # Skip None values
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

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], data: Dict[str, Any], context: Dict[str, Any] | None = None) -> T:
        """
        Reconstruct a component from a dictionary.

        Args:
            data: Dictionary representation of the component
            context: Optional context containing derived parameters (e.g., building_props)

        Returns:
            Reconstructed component instance
        """
        pass

    @classmethod
    def _get_init_args(
        cls,
        data: Dict[str, Any],
        context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Build kwargs dict for __init__ from serialized data and context.

        Merges serialized data with context parameters.

        Args:
            data: Serialized data dictionary
            context: Optional context with derived parameters

        Returns:
            kwargs dictionary for __init__
        """
        context = context or {}
        kwargs = {}

        # Get __init__ signature
        sig = inspect.signature(cls.__init__)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # Check if value is in serialized data
            if param_name in data:
                kwargs[param_name] = data[param_name]
            # Check if value is in context
            elif param_name in context:
                kwargs[param_name] = context[param_name]
            # Check if parameter has a default value
            elif param.default is not inspect.Parameter.empty:
                # Use default from signature
                pass
            else:
                # Required parameter is missing
                logger.warning(
                    "Missing required parameter '%s' for %s",
                    param_name, cls.__name__
                )

        return kwargs
