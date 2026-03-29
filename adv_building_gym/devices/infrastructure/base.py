import logging
from collections import OrderedDict
from typing import Any, ClassVar, Dict, Set, Type, TypeVar

from adv_building_gym.utils import EnvSyncInterface
from adv_building_gym.utils.serializable import Serializable, ComponentRegistry

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Infrastructure')


class Infrastructure(EnvSyncInterface, Serializable):
    """Base class for infrastructure components in the building environment."""

    # Parameters derived from context (building_props, control_step)
    _context_params: ClassVar[Set[str]] = set()

    # Internal state - never serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'row_offset'}

    def __init__(self,
                name: str,
                max_power_kW: float
                ) -> None:
        super().__init__()

        self.name = name
        self.max_power_kW = max_power_kW  # ~ rated power capacity

    @property
    def max_consumption_kW(self) -> float:
        """Maximum power this component can draw from the grid (kW).

        Defaults to max_power_kW (consumption-only device).
        Override in subclasses with different power flow directions.
        """
        return self.max_power_kW

    @property
    def max_export_kW(self) -> float:
        """Maximum power this component can export to the grid (kW).

        Defaults to 0 (consumption-only device).
        Override in subclasses that can produce or discharge.
        """
        return 0.0

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation and action spaces. Implement in derived classes."""
        return state_spaces, action_spaces

    def set_target(self, target: float) -> None:
        """Set target for infrastructure component."""
        pass

    def exec_action(self, actions: Dict, states: Dict, info: dict | None = None) -> None:
        """Execute action of the infrastructure.

        Args:
            actions: Action dict keyed by component action name.
            states: Observable state dict; treat as immutable during action execution.
            info: Shared dict for inter-component data that is not part of
                the observation space (e.g., EV schedule parameters).
        """
        pass

    def update_state(self, states: Dict, info: dict | None = None) -> None:
        """Update state based on current iteration.

        Subclasses must call ``super().update_state(states, info)`` so
        that base-class bookkeeping (power bound publication) runs.

        **Note**: Called after ``exec_action`` to update observable states,
        and only to update them, not to perform actions.

        Args:
            states: Observable state dict (agent-visible).
            info: Shared dict for inter-component data that is not part of
                the observation space.
        """
        info = self._publish_power_bounds(info)

    def reset(self, states: Dict, info: dict | None = None) -> None:
        """Populate initial state at episode start (after data reloads).

        Called once per episode instead of update_state() during reset().
        The default delegates to update_state(); subclasses can override
        for reset-specific initialisation.
        """
        self.update_state(states, info)

    def _publish_power_bounds(self, info: dict | None = None) -> dict:
        """Accumulate this component's directional power bounds into *info*.

        Creates a new dict if *info* is ``None`` so the bounds are always
        available via the returned value.
        """
        if info is None:
            info = {}
        info["max_consumption_kW"] = info.get("max_consumption_kW", 0.0) + self.max_consumption_kW
        info["max_export_kW"] = info.get("max_export_kW", 0.0) + self.max_export_kW
        return info

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption in kW.

        Default implementation: extracts action for this component and scales by max_power_kW.
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
