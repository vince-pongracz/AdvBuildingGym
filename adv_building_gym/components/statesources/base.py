import logging
from typing import ClassVar, Optional, Set

import pandas as pd

from adv_building_gym.core.env_sync import EnvSync
from adv_building_gym.components.registry import Serializable
from adv_building_gym._common.lifecycle import ReloadObserver
from .csv_loader import CsvLoader

logger = logging.getLogger(__name__)


class StateSource(Serializable):
    """Base class for data sources in the building environment.

    Composition over inheritance:
      * Synchronisation state lives in ``self.sync`` (an ``EnvSync`` instance)
        exposed via pass-through properties (``iteration``, ``row_offset``,
        ``effective_index``, ``synchronise``).
      * File-backed time series live in ``self.loader`` (a ``CsvLoader``)
        when present. Sources that don't load from CSV (e.g.
        ``OperatorEnergyControl``) leave ``self.loader`` as ``None`` and
        inherit the base's default ``reload`` which raises ``TypeError``.

    Pass-through properties ``ts`` / ``ds_path`` / ``is_new_data_source`` keep
    subclass call sites (``self.ts``, ``self.is_new_data_source``) unchanged.
    """

    # Parameters derived from context (building_props, control_step)
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state - never serialize
    _exclude_params: ClassVar[Set[str]] = set()

    def __init__(self,
                name: str,
                control_step: float = 300.0,
                ) -> None:
        super().__init__()

        self.sync = EnvSync()
        self.name = name
        self.control_step = control_step  # Control timestep in seconds
        # CSV-backed subclasses assign ``self.loader = CsvLoader(ds_path,
        # on_reload=self._run_post_load)`` after setting their own attributes.
        self.loader: Optional[CsvLoader] = None

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

    # ----- CsvLoader pass-throughs (composition; None for non-CSV sources) -----
    @property
    def ts(self) -> Optional[pd.DataFrame]:
        return self.loader.ts if self.loader is not None else None

    @property
    def ds_path(self) -> Optional[str]:
        return self.loader.ds_path if self.loader is not None else None

    @property
    def is_new_data_source(self) -> bool:
        """True when the current ds_path differs from the last processed one.

        Used by subclasses to guard one-time diagnostics (e.g. NaN warnings).
        Always ``False`` for sources without a loader.
        """
        return self.loader.is_new_data_source if self.loader is not None else False

    def _post_load_data_processing(self) -> None:
        """Override to re-run post-processing after a new CSV is loaded.

        Called from ``_run_post_load`` after any ``ReloadObserver`` notification
        and after the underlying ``CsvLoader`` has populated ``self.ts``.
        Subclasses that normalise columns, cache scalars, or parse events from
        the CSV put that logic here. Use ``self.is_new_data_source`` to gate
        one-time diagnostics so they fire only when the file actually changes.
        """

    def _run_post_load(self) -> None:
        """Notify reload observers, then run subclass post-processing.

        Wired as the ``on_reload`` callback of the source's ``CsvLoader`` so
        it fires automatically after every successful read. Any mixin
        satisfying the ``ReloadObserver`` protocol (i.e. defining
        ``on_reload``) is notified before subclass post-processing — e.g.
        ``Forecastable`` drops its cached column views here.
        """
        if isinstance(self, ReloadObserver):
            self.on_reload()
        self._post_load_data_processing()

    def reload(self, ds_path: str) -> None:
        """Load a new time-series file without recreating this StateSource.

        Delegates to ``self.loader``. Sources without a loader (e.g.
        ``OperatorEnergyControl``) raise — they are not file-backed.
        """
        if self.loader is None:
            raise TypeError(
                f"{type(self).__name__} '{self.name}' is not a file-backed source "
                "and does not support reload()."
            )
        self.loader.reload(ds_path)
        logger.debug("StateSource '%s' reloaded from %s", self.name, ds_path)

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
