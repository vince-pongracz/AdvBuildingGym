import logging
from collections import OrderedDict
from typing import Any, ClassVar, Dict, Literal, Set

from adv_building_gym.core.env_sync import EnvSync
from adv_building_gym.components.context_emitter import ContextEmitter

logger = logging.getLogger(__name__)

PowerFlow = Literal["consumer", "generator", "bidirectional"]


class Infrastructure(ContextEmitter):
    """Base infrastructure component.

    Sync state lives in ``self.sync`` (``EnvSync``), exposed via pass-through
    properties so ``self.iteration`` / ``self.synchronise`` keep working.
    """

    # Parameters derived from context (building_props, control_step)
    _context_params: ClassVar[Set[str]] = set()

    # Internal state - never serialize
    _exclude_params: ClassVar[Set[str]] = set()

    # Power-flow direction; every subclass MUST set it (enforced in
    # __init_subclass__). Not an __init__ arg, so not serialised.
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

        self.sync = EnvSync()
        self.name = name
        self.max_power_kW = max_power_kW  # ~ rated power capacity

    # ----- EnvSync pass-throughs (composition) -----
    @property
    def iteration(self) -> int:
        return self.sync.iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        self.sync.iteration = value

    @property
    def row_offset(self) -> int:
        return self.sync.row_offset

    @row_offset.setter
    def row_offset(self, value: int) -> None:
        self.sync.row_offset = value

    @property
    def effective_index(self) -> int:
        return self.sync.effective_index

    def synchronise(self, iteration: int, row_offset: int | None = None) -> None:
        self.sync.synchronise(iteration, row_offset)

    @property
    def max_consumption_kW(self) -> float:
        """Max power possibly drawn from grid (kW), from POWER_FLOW. Override for runtime-dependent bounds."""
        return self.max_power_kW if self.POWER_FLOW in ("consumer", "bidirectional") else 0.0

    @property
    def max_production_kW(self) -> float:
        """Max power possiblyexported to grid (kW), from POWER_FLOW. Override for runtime-dependent bounds."""
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
        """Execute the component's action.

        actions: action dict by component name. states: observable state
        (treat immutable here). info: shared inter-component data (not in obs).
        """
        pass

    def update_state(self, states: Dict, info: dict | None = None) -> None:
        """Update observable state (after exec_action; no actions here).

        Subclasses must call ``super().update_state(states, info)`` for
        base bookkeeping (power-bound publication).
        """
        info = self._publish_power_bounds(info)

    def reset(self, states: Dict, info: dict | None = None) -> None:
        """Populate initial state at episode start (after data reloads).

        Called once per episode in place of update_state(); default delegates to it.
        """
        self.update_state(states, info)

    def _publish_power_bounds(self, info: dict | None = None) -> dict:
        """Accumulate this component's directional power bounds into *info* (new dict if None)."""
        if info is None:
            info = {}
        info["max_consumption_kW"] = info.get("max_consumption_kW", 0.0) + self.max_consumption_kW
        info["max_production_kW"] = info.get("max_production_kW", 0.0) + self.max_production_kW
        return info

    def get_raw_values(self) -> dict[str, float]:
        """Raw (unnormalised) physical values for logging; override to populate."""
        return {}

    def get_E(self, actions: Dict) -> tuple[float, float]:
        """Electric energy (kW) as (production, consumption). Default 0; override per component."""
        # default: no action -> no power
        return 0.0, 0.0

