import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..forecastable import Forecastable
from adv_building_gym.utils.serializable import ComponentRegistry
from adv_building_gym.utils.normalisation import Normalisation, normalise_with_scale_factor

logger = logging.getLogger(__name__)

# CSV column produced by preprocessing/hh_consumption/extract_hh_consumption.py
SOURCE_COLUMN: str = "hh_consumption_kW"

# Normalised column name added by _post_load_data_processing
NORM_COLUMN: str = "desired_energy_need_norm"


class DesiredUserEnergyNeed(StateSource, Forecastable):
    """Data source for desired user energy need information.

    When a CSV is provided (via ``ds_path``), reads the ``hh_consumption_kW``
    column produced by the WPuQ household consumption preprocessing pipeline,
    normalises it to [0, 1] using min-max scaling, and exposes it as the
    ``desired_energy_need`` observation.

    state key: desired_energy_need
    Definition desired_energy_need:
    - positive: user consumes E.
    - negative: no meaning
    - zero: user does not need E.
    """

    # consumption_max is derived from data, don't serialize
    _exclude_params: ClassVar[Set[str]] = {'consumption_max'}

    def __init__(self, name: str, ds_path: str | None = None,
                normalise: Normalisation | str | None = Normalisation.MIN_MAX_SCALING) -> None:
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

    def setup_spaces(self, state_spaces: OrderedDict, action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation spaces for desired user energy need."""

        if "s_desired_energy_need" not in state_spaces.keys():
            state_spaces["s_desired_energy_need"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Raw maximum household consumption (kW) — changes only when a new
        # data variant is loaded.
        if "ctxt_hh_consumption_max" not in state_spaces.keys():
            state_spaces["ctxt_hh_consumption_max"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        if "raw_sim_hour" not in state_spaces.keys():
            state_spaces["raw_sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,), dtype=np.float32)

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
        # Raw maximum consumption (kW) — constant within an episode.
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


# Register DesiredUserEnergyNeed with the component registry
ComponentRegistry.register('statesource', DesiredUserEnergyNeed)
