import logging
from typing import Any, ClassVar, Dict, Set, Type, TypeVar

import pandas as pd

from adv_building_gym.utils import EnvSyncInterface
from adv_building_gym.config.utils.serializable import Serializable, ComponentRegistry

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='StateSource')


class StateSource(EnvSyncInterface, Serializable):
    """Base class for data sources in the building environment."""

    # Parameters derived from context (building_props, control_step)
    _context_params: ClassVar[Set[str]] = set()

    # Internal state - never serialize (ts is loaded from ds_path)
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'ts'}

    def __init__(self,
                 name: str,
                 ds_path: str | None = None,
                 ) -> None:
        super().__init__()

        self.name = name
        self.ds_path = ds_path  # Store path for serialization
        if ds_path is not None:
            self.ts = pd.read_csv(ds_path)
            """Time series"""
        else:
            self.ts = None

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        """Setup observation and action spaces. Implement in derived classes."""
        return state_spaces, action_spaces

    def update_state(self, states) -> None:
        """Update state based on current iteration. Implement in derived classes."""
        pass

    @classmethod
    def from_dict(
        cls: Type[T],
        data: Dict[str, Any],
        context: Dict[str, Any] | None = None
    ) -> T:
        """
        Reconstruct a StateSource from a dictionary.

        Uses the ComponentRegistry to find the correct class by name,
        then constructs it with serialized data merged with context.

        Args:
            data: Dictionary containing 'class' key and constructor parameters
            context: Optional context with derived parameters (e.g., K, mC, timestep)

        Returns:
            Reconstructed StateSource instance
        """
        class_name = data.get('class')
        if class_name is None:
            raise ValueError("Missing 'class' key in statesource data")

        # Get the actual class from registry
        source_class = ComponentRegistry.get('statesource', class_name)

        # Build kwargs from data and context
        kwargs = source_class._get_init_args(data, context)

        return source_class(**kwargs)
