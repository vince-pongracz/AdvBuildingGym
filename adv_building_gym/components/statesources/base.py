import logging
from typing import ClassVar, Optional, Set

import pandas as pd

from adv_building_gym.core.env_sync import EnvSync
from adv_building_gym.components.registry import Serializable
from adv_building_gym._common.lifecycle import ReloadObserver
from .csv_loader import CsvLoader

logger = logging.getLogger(__name__)


class StateSource(Serializable):
    """Base class for env data sources.

    Sync state lives in ``self.sync`` (``EnvSync``);
    CSV-backed series in
    ``self.loader`` (``CsvLoader``), ``None`` for non-CSV sources (whose default
    ``reload`` raises). Pass-through properties (``iteration``/``row_offset``/
    ``effective_index``/``ts``/``ds_path``/``is_new_data_source``) keep call sites unchanged.
    """

    # Parameters derived from context (building_props, control_step)
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state - never serialize
    _exclude_params: ClassVar[Set[str]] = set()

    # When update_state runs within step():
    #   "exogenous" (default): external series, advanced AFTER the reward (so rewards
    #       see the observed values).
    #   "endogenous": within-step physics on action-affected state (e.g. heat loss),
    #       run BEFORE the reward under the observed exogenous row.
    UPDATE_PHASE: ClassVar[str] = "exogenous"

    def __init__(self,
                name: str,
                control_step: float = 300.0,
                ) -> None:
        super().__init__()

        self.sync = EnvSync()
        self.name = name
        self.control_step = control_step  # Control timestep in seconds
        # CSV-backed subclasses set ``self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)``.
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
        """True when ds_path differs from the last processed (guards one-time diagnostics); False without a loader."""
        return self.loader.is_new_data_source if self.loader is not None else False

    def _post_load_data_processing(self) -> None:
        """Override for post-load processing (normalise columns, cache scalars, parse events).

        Runs after ``CsvLoader`` populates ``self.ts``; gate one-time diagnostics with
        ``self.is_new_data_source``.
        """

    def _run_post_load(self) -> None:
        """Notify ReloadObserver mixins (e.g. Forecastable), then run subclass post-processing.

        Wired as the CsvLoader ``on_reload`` callback, firing after every read.
        """
        if isinstance(self, ReloadObserver):
            self.on_reload()
        self._post_load_data_processing()

    def reload(self, ds_path: str) -> None:
        """Load a new time-series file (delegates to ``self.loader``); non-file-backed sources raise."""
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
        """Update observable state for the current iteration (implement in subclasses)."""
        pass

    def reset(self, states, info: dict | None = None) -> None:
        """Populate initial state at episode start (after reloads).

        Once per episode in place of update_state(); default delegates to it.
        """
        self.update_state(states, info)

    def get_raw_values(self) -> dict[str, float]:
        """Raw (unnormalised) physical values for logging; override to populate."""
        return {}
