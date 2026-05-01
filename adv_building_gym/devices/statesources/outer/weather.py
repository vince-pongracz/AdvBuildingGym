import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.utils.serializable import ComponentRegistry
from adv_building_gym.utils.normalisation import Normalisation, normalise_with_scale_factor

logger = logging.getLogger(__name__)


class WeatherDataSource(StateSource):
    """WeatherDataSource"""

    # normalise is an enum, need special handling for serialization
    _context_params: ClassVar[Set[str]] = {'control_step'}
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'ts',
        'temp_abs_max', 'temp_out_raw',
        'wind_speed_abs_max', 'wind_speed_raw',
        'sun_shine_raw',
    }

    def __init__(self, name: str, ds_path: str | None = None,
                normalise: Normalisation | str | None = Normalisation.ABS_MIN_MAX_SCALING) -> None:
        super().__init__(name, ds_path)

        self.normalise = Normalisation.init(normalise)  # Store for serialization
        # Raw values for get_raw_values() — updated each step
        self.temp_out_raw: float = 0.0
        self.wind_speed_raw: float = 0.0
        # Solar irradiance in J/cm² (DWD/Zenodo unit; see preprocessing).
        self.sun_shine_raw: float = 0.0

        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            self._run_post_load()
        else:
            logger.debug("No initial data file for '%s', data source will be assigned by DataCombinator", name)

    def _post_load_data_processing(self) -> None:
        """Normalise weather columns after CSV load / reload.

        Data cleaning (sentinel replacement, NaN handling, column aliasing)
        is handled by the preprocessing scripts. This method only validates
        that the data is clean and applies runtime normalisation.
        """
        # Validate and clean data; only warn when a new file is loaded (not
        # on every reload of the same file, which happens each episode reset).
        weather_cols = ["temp_amb", "sun_shine", "avg_wind_speed"]
        for col in weather_cols:
            if col in self.ts.columns:
                n_nan = int(self.ts[col].isna().sum())
                if n_nan > 0:
                    if self.is_new_data_source:
                        logger.warning(
                            "WeatherDataSource '%s': %d NaN in '%s' — "
                            "check preprocessing. Filling with 0.",
                            self.name, n_nan, col,
                        )
                    self.ts[col] = self.ts[col].fillna(0)

        if "sun_shine" not in self.ts.columns and "direct_sun_shine" in self.ts.columns:
            if self.is_new_data_source:
                logger.warning(
                    "WeatherDataSource '%s': 'sun_shine' column missing, "
                    "falling back to 'direct_sun_shine' — check preprocessing.",
                    self.name,
                )
            self.ts["sun_shine"] = self.ts["direct_sun_shine"]

        # Normalise raw columns and derive scale factors so downstream
        # components can convert between raw and normalised values.
        cols = {
            "temp_amb": ("s_temp_out_norm", "temp_abs_max"),
            "sun_shine": ("s_solar_irradiance_norm", None),
            "avg_wind_speed": ("s_avg_wind_speed_norm", "wind_speed_abs_max"),
        }

        for raw_col, (norm_col, scale_attr) in cols.items():
            if raw_col in self.ts.columns:
                normalised, scale_factor = normalise_with_scale_factor(self.ts[raw_col], self.normalise)
                self.ts[norm_col] = normalised
                if scale_attr is not None:
                    setattr(self, scale_attr, scale_factor)


    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict
                    ) -> tuple[OrderedDict, OrderedDict]:

        if "s_temp_out_norm" not in state_spaces.keys():
            state_spaces["s_temp_out_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "s_solar_irradiance_norm" not in state_spaces.keys():
            state_spaces["s_solar_irradiance_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_avg_wind_speed_norm" not in state_spaces.keys():
            state_spaces["s_avg_wind_speed_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Raw scale factors — set once when data is loaded, not every step.
        # The policy can use these to reconstruct physical units from
        # normalised observations (e.g. temp_out_raw = temp_out_norm * temp_abs_max).
        if "ctxt_temp_abs_max" not in state_spaces.keys():
            state_spaces["ctxt_temp_abs_max"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        if "ctxt_wind_speed_abs_max" not in state_spaces.keys():
            state_spaces["ctxt_wind_speed_abs_max"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        if "raw_sim_hour" not in state_spaces.keys():
            state_spaces["raw_sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,),
                                            dtype=np.float32)


        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        if self.ts is None:
            raise RuntimeError(
                f"WeatherDataSource '{self.name}': no CSV loaded. The DataCombinator "
                "must push a weather variant before update_state is called."
            )
        row = self.ts.iloc[min(self.effective_index, len(self.ts) - 1)]
        temp_out_norm = float(row["s_temp_out_norm"])
        self.temp_out_raw = float(row["temp_amb"])
        solar_irradiance_norm = float(row.get("s_solar_irradiance_norm", 0.0))
        avg_wind_speed_norm = float(row.get("s_avg_wind_speed_norm", 0.0))
        self.wind_speed_raw = float(row.get("avg_wind_speed", 0.0))
        self.sun_shine_raw = float(row.get("sun_shine", 0.0))

        states["s_temp_out_norm"][0] = np.float32(temp_out_norm)
        states["s_solar_irradiance_norm"][0] = np.float32(solar_irradiance_norm)
        states["s_avg_wind_speed_norm"][0] = np.float32(avg_wind_speed_norm)

        # Raw scale factors — constant within an episode, change only when
        # a new data variant is loaded (via _post_load_data_processing).
        states["ctxt_temp_abs_max"][0] = np.float32(self.temp_abs_max)
        states["ctxt_wind_speed_abs_max"][0] = np.float32(self.wind_speed_abs_max)

    def get_raw_values(self) -> dict[str, float]:
        return {
            "raw_temp_out": self.temp_out_raw,
            "raw_wind_speed": self.wind_speed_raw,
            "raw_solar_irradiance": self.sun_shine_raw,
        }

    def _get_serialize_value(self, param_name: str, value):
        """Handle enum serialization for normalise parameter."""
        if param_name == 'normalise' and isinstance(value, Normalisation):
            return value.value  # Serialize as string
        return super()._get_serialize_value(param_name, value)


# Register WeatherDataSource with the component registry
ComponentRegistry.register('statesource', WeatherDataSource)