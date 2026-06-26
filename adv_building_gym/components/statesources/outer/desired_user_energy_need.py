import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..forecastable import Forecastable
from ..csv_lookahead import CsvLookahead
from ..reloadable import CsvReloadable
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.normalisation import Normalisation, normalise_with_scale_factor

logger = logging.getLogger(__name__)

# CSV column produced by preprocessing/hh_consumption/extract_hh_consumption.py
SOURCE_COLUMN: str = "hh_consumption_kW"

# Normalised column name added by _post_load_data_processing
NORM_COLUMN: str = "desired_energy_need_norm"


class DesiredUserEnergyNeed(StateSource, Forecastable, CsvLookahead, CsvReloadable):
    """Desired user energy need from the ``hh_consumption_kW`` CSV column.

    Abs-min-max normalised to [0, 1], exposed as ``s_desired_energy_need``
    (1 = peak load, 0 = no need; negative is meaningless). For non-negative
    consumption the scale factor equals the raw max, so ``raw_kW = norm * scale``
    holds exactly and ``ctxt_hh_consumption_max`` is the genuine peak in kW —
    consumed by ``HouseholdEnergyConsumers`` to recover physical kW.
    """

    # consumption_max is derived from data, don't serialize
    _exclude_params: ClassVar[Set[str]] = {'consumption_max'}

    def __init__(self, name: str, ds_path: str | None = None,
                normalise: Normalisation | str | None = Normalisation.MAX_ABS_SCALING) -> None:
        super().__init__(name=name)

        self.normalise = Normalisation.init(normalise)

        self.consumption_max: float = 1.0

        self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)
        if ds_path is not None:
            logger.info("Use data file: %s", ds_path)

    def _post_load_data_processing(self) -> None:
        """Normalise the hh_consumption_kW column and cache the raw maximum."""
        if SOURCE_COLUMN not in self.ts.columns:
            raise ValueError(
                f"DesiredUserEnergyNeed '{self.name}': CSV '{self.ds_path}' has no "
                f"'{SOURCE_COLUMN}' column. Found: {list(self.ts.columns)}"
            )

        self.ts[NORM_COLUMN], self.consumption_max = normalise_with_scale_factor(self.ts[SOURCE_COLUMN], self.normalise)

        # Only the normalised column is read each step (the raw kW peak is cached in
        # consumption_max); drop the raw source column and any other CSV columns.
        self._keep_ts_columns({NORM_COLUMN})

    def setup_spaces(self, state_spaces: OrderedDict, action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation spaces for desired user energy need."""

        if "s_desired_energy_need" not in state_spaces.keys():
            state_spaces["s_desired_energy_need"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Max household consumption (kW) — changes per data variant.
        if "ctxt_hh_consumption_max" not in state_spaces.keys():
            state_spaces["ctxt_hh_consumption_max"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        """Update desired energy need state based on current iteration."""
        if self.ts is None:
            raise RuntimeError(
                f"DesiredUserEnergyNeed '{self.name}': no CSV loaded. The DataCombinator "
                "must push a user_energy_need variant before update_state is called."
            )
        row = self.ts.iloc[min(self.effective_index, len(self.ts) - 1)]
        desired_energy = float(row[NORM_COLUMN])

        states["s_desired_energy_need"][0] = np.float32(desired_energy)
        # max consumption (kW) — constant per episode
        states["ctxt_hh_consumption_max"][0] = np.float32(self.consumption_max)

    def forecast_keys(self) -> tuple[str, ...]:
        return ("s_fc_desired_energy_need",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        if self.ts is None:
            return {"s_fc_desired_energy_need": [0.0] * len(selected_future_steps)}
        return {
            "s_fc_desired_energy_need": self._csv_forecast(self.ts, self.effective_index, NORM_COLUMN, selected_future_steps),
        }

    @property
    def consumption_max_raw(self) -> float:
        """Raw (unnormalised) maximum consumption in kW."""
        return float(self.consumption_max)

    def _get_serialize_value(self, param_name: str, value):
        """Handle enum serialization for normalise parameter."""
        if param_name == 'normalise' and isinstance(value, Normalisation):
            return value.value
        return super()._get_serialize_value(param_name, value)


# register with ComponentRegistry
ComponentRegistry.register('statesource', DesiredUserEnergyNeed)
