import logging
from typing import Any, ClassVar, Dict, Set, Type, TypeVar

from adv_building_gym.utils import EnvSyncInterface
from adv_building_gym.config.utils.serializable import Serializable, ComponentRegistry

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Infrastructure')


class Infrastructure(EnvSyncInterface, Serializable):
    """Base class for infrastructure components in the building environment."""

    # Parameters derived from context (building_props, control_step)
    _context_params: ClassVar[Set[str]] = set()

    # Internal state - never serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration'}

    def __init__(self,
                 name: str,
                 Q_electric_max: float
                 ) -> None:
        super().__init__()

        self.name = name
        self.Q_electric_max = Q_electric_max  # ~ power consumption max

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        """Setup observation and action spaces. Implement in derived classes."""
        return state_spaces, action_spaces

    def set_target(self, target: float) -> None:
        """Set target for infrastructure component."""
        pass

    def exec_action(self, actions, states) -> None:
        """Execute action of the infrastructure.

        Args:
            actions (Dict): contains actions
            states (Dict): should be treated as immutable, holds information for the action execution
        """
        pass

    def update_state(self, states: Dict) -> None:
        """
        Update state based on current iteration. Implement in derived classes.
        
        **Note**: Called after `exec_action` to update observable states, 
        and only to update them, not to perform actions.
        """
        pass

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption in kW.

        Default implementation: extracts action for this component and scales by Q_electric_max.
        Override in derived classes for more complex calculations.

        Args:
            actions: Dictionary containing all actions

        Returns:
            Electric energy consumption in kW
        """
        # Default: return 0 if no action found
        return 0.0

    @classmethod
    def from_dict(
        cls: Type[T],
        data: Dict[str, Any],
        context: Dict[str, Any] | None = None
    ) -> T:
        """
        Reconstruct an Infrastructure from a dictionary.

        Uses the ComponentRegistry to find the correct class by name,
        then constructs it with serialized data merged with context.

        Args:
            data: Dictionary containing 'class' key and constructor parameters
            context: Optional context with derived parameters (e.g., K, mC from building_props)

        Returns:
            Reconstructed Infrastructure instance
        """
        class_name = data.get('class')
        if class_name is None:
            raise ValueError("Missing 'class' key in infrastructure data")

        # Get the actual class from registry
        infra_class = ComponentRegistry.get('infrastructure', class_name)

        # Build kwargs from data and context
        kwargs = infra_class._get_init_args(data, context)

        return infra_class(**kwargs)
