import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Set, Type, TypeVar

import pandas as pd

from adv_building_gym.utils import EnvSyncInterface
from adv_building_gym.utils.serializable import Serializable, ComponentRegistry

logger = logging.getLogger(__name__)

# Project root directory (three levels up: base.py -> statesources -> devices -> adv_building_gym -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

T = TypeVar('T', bound='StateSource')


class StateSource(EnvSyncInterface, Serializable):
    """Base class for data sources in the building environment."""

    # Parameters derived from context (building_props, control_step)
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state - never serialize (ts is loaded from ds_path)
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'row_offset', 'ts', '_last_processed_ds_path'}

    def __init__(self,
                name: str,
                ds_path: str | None = None,
                control_step: float = 300.0,
                ) -> None:
        super().__init__()

        self.name = name
        self.ds_path = ds_path  # Store original path for serialization
        self.control_step = control_step  # Control timestep in seconds
        self._last_processed_ds_path: str | None = None  # Tracks which file was last post-processed
        if ds_path is not None:
            resolved = Path(ds_path)
            if not resolved.is_absolute():
                resolved = _PROJECT_ROOT / resolved
            """Time series"""
            self.ts = pd.read_csv(resolved)
        else:
            self.ts = None

    @property
    def is_new_data_source(self) -> bool:
        """True when the current ds_path differs from the last processed one.

        Subclasses can check this in ``_post_load_data_processing`` to decide
        whether to emit one-time warnings (e.g. NaN validation).  The flag is
        updated automatically after ``_post_load_data_processing`` returns.
        """
        return self.ds_path != self._last_processed_ds_path

    def _post_load_data_processing(self) -> None:
        """Override to re-run post-processing after a new CSV is loaded.

        Called both at the end of __init__ (via subclass constructors) and
        after reload().  Subclasses that normalise columns, cache scalars, or
        parse events from the CSV should put that logic here.

        Use ``self.is_new_data_source`` to guard one-time diagnostics
        (e.g. NaN warnings) so they only fire when the file actually changes.
        """

    def _run_post_load(self) -> None:
        """Run subclass post-processing and update the data-source tracker."""
        self._post_load_data_processing()
        self._last_processed_ds_path = self.ds_path

    def reload(self, ds_path: str) -> None:
        """Load a new time-series file without recreating this StateSource.

        Replaces the underlying DataFrame, re-runs subclass post-processing
        via _post_load(), and resets the iteration counter so the next episode
        reads from row 0.

        Relative paths are resolved against the project root so that Ray
        worker processes (whose CWD may differ) can still find the files.
        """
        resolved = Path(ds_path)
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        self.ds_path = ds_path
        self.ts = pd.read_csv(resolved)
        self._run_post_load()
        logger.debug("StateSource '%s' reloaded from %s", self.name, resolved)

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Setup observation and action spaces. Implement in derived classes."""
        return state_spaces, action_spaces

    def update_state(self, states, info: dict | None = None) -> None:
        """Update state based on current iteration. Implement in derived classes.

        Args:
            states: Observable state dict (agent-visible).
            info: Shared dict for inter-component data that is not part of
                the observation space (e.g., scale factors, EV schedule).
        """
        pass

    def reset(self, states, info: dict | None = None) -> None:
        """Populate initial state at episode start (after data reloads).

        Called once per episode instead of update_state() during reset().
        The default implementation delegates to update_state(); subclasses
        can override to apply reset-specific initialisation (e.g. seeding
        temp_in_norm from the desired setpoint).
        """
        self.update_state(states, info)

    def get_raw_values(self) -> dict[str, float]:
        """Return raw (unnormalised) physical values for logging.

        Override in subclasses that track raw values (e.g. raw temperature).
        Default returns an empty dict.
        """
        return {}

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
