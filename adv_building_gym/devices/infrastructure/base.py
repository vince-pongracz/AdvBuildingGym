import logging
from collections import OrderedDict
from typing import Any, ClassVar, Dict, Literal, Set, Type, TypeVar

from adv_building_gym.utils import EnvSyncInterface
from adv_building_gym.utils.serializable import Serializable, ComponentRegistry

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Infrastructure')

PowerFlow = Literal["consumer", "generator", "bidirectional"]


class Infrastructure(EnvSyncInterface, Serializable):
    """Base class for infrastructure components in the building environment."""

    # Parameters derived from context (building_props, control_step)
    _context_params: ClassVar[Set[str]] = set()

    # Internal state - never serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'row_offset'}

    # Class-level declaration of power-flow direction. Every concrete subclass
    # MUST set this; enforced in __init_subclass__. Not an __init__ arg, so it
    # is not serialised into YAML.
    POWER_FLOW: ClassVar[PowerFlow | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if (cls.POWER_FLOW is None or
                cls.POWER_FLOW not in ("consumer", "generator", "bidirectional")):
            raise TypeError(
                f"{cls.__name__} must declare {cls.__name__}.POWER_FLOW as one of 'consumer', 'generator', or 'bidirectional'."
            )

    def __init__(self, name: str, max_power_kW: float) -> None:
        super().__init__()

        self.name = name
        self.max_power_kW = max_power_kW  # ~ rated power capacity

    @property
    def max_consumption_kW(self) -> float:
        """Maximum power possibly DRAWN from the grid (kW).

        Derived from POWER_FLOW. Override only when the bound depends on
        runtime state (e.g., a flag toggling a direction).
        """
        return self.max_power_kW if self.POWER_FLOW in ("consumer", "bidirectional") else 0.0

    @property
    def max_production_kW(self) -> float:
        """Maximum power possibly EXPORTED to the grid (kW).

        Derived from POWER_FLOW. Override only when the bound depends on
        runtime state (e.g., a flag toggling a direction).
        """
        return self.max_power_kW if self.POWER_FLOW in ("generator", "bidirectional") else 0.0

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
        info["max_production_kW"] = info.get("max_production_kW", 0.0) + self.max_production_kW
        return info

    def get_raw_values(self) -> dict[str, float]:
        """Return raw (unnormalised) physical values for logging.

        Override in subclasses that track raw values.
        Default returns an empty dict.
        """
        return {}

    def get_penalisable_consumption(self, actions: Dict, states: Dict) -> float:
        """Power (kW) that should count toward the energy consumption penalty.

        Override to exempt necessary consumption (e.g. charging below target
        SoC) or non-controllable load.  Default: all consumption is penalisable.
        """
        E_production, E_consumption = self.get_E(actions)
        return E_consumption


    def get_E(self, actions: Dict) -> tuple[float, float]:
        """Get current electric energy consumption in kW.

        Default implementation: extracts action for this component and scales by max_power_kW.
        Override in derived classes for more complex calculations.

        Args:
            actions: Dictionary containing all actions

        Returns:
            float1 -- production
            
            float2 -- consumption
        """
        # Default: return 0 if no action found
        return 0.0, 0.0

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
