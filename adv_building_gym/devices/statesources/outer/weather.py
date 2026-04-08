import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.utils.serializable import ComponentRegistry
from adv_building_gym.utils.normalisation import Normalisation, normalise_series

logger = logging.getLogger(__name__)


class WeatherDataSource(StateSource):
    """WeatherDataSource"""

    # normalise is an enum, need special handling for serialization
    _context_params: ClassVar[Set[str]] = {'control_step', 'temp_abs_max'}
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'ts', '_fixed_temp_abs_max', 'wind_speed_abs_max'}

    def __init__(self, name: str, ds_path: str | None = None,
                normalise: Normalisation | str | None = Normalisation.ABS_MIN_MAX_SCALING,
                temp_abs_max: float | None = None) -> None:
        super().__init__(name, ds_path)

        self.normalise = Normalisation.init(normalise)  # Store for serialization
        self.temp_out_raw: float = 0.0  # Raw outdoor temperature (°C)
        # Fixed scale factor from config (max(|temp_min|, |temp_max|)).
        # When set, temperature normalisation uses this instead of the
        # data-derived value, ensuring consistent scaling across datasets.
        self._fixed_temp_abs_max: float | None = temp_abs_max
        self.temp_abs_max: float = temp_abs_max if temp_abs_max is not None else 1.0
        self.wind_speed_abs_max: float = 1.0  # Derived from data in _post_load_data_processing

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

        # Normalise raw columns for the observation space
        cols = {
            "temp_amb": "temp_out_norm",
            "sun_shine": "solar_irradiance_norm",
            "avg_wind_speed": "avg_wind_speed_norm",
        }

        for raw_col, norm_col in cols.items():
            if raw_col in self.ts.columns:
                if raw_col == "temp_amb" and self._fixed_temp_abs_max is not None:
                    # Use fixed config range for temperature normalisation
                    self.ts[norm_col] = self.ts[raw_col] / self._fixed_temp_abs_max
                else:
                    self.ts[norm_col] = normalise_series(self.ts[raw_col], self.normalise)

        # Use fixed scale factor when provided, otherwise derive from data
        if self._fixed_temp_abs_max is not None:
            self.temp_abs_max = self._fixed_temp_abs_max
        elif "temp_amb" in self.ts.columns:
            self.temp_abs_max = float(self.ts["temp_amb"].abs().max())
        else:
            self.temp_abs_max = 1.0

        # Wind speed scale factor — derived from data (always non-negative)
        if "avg_wind_speed" in self.ts.columns:
            self.wind_speed_abs_max = float(self.ts["avg_wind_speed"].abs().max())
            if self.wind_speed_abs_max == 0.0:
                self.wind_speed_abs_max = 1.0
        else:
            self.wind_speed_abs_max = 1.0


    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict
                    ) -> tuple[OrderedDict, OrderedDict]:
        if "temp_out_norm" not in state_spaces.keys():
            state_spaces["temp_out_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "solar_irradiance_norm" not in state_spaces.keys():
            state_spaces["solar_irradiance_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "avg_wind_speed_norm" not in state_spaces.keys():
            state_spaces["avg_wind_speed_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,),
                                            dtype=np.float32)


        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        if self.ts is not None:
            row = self.ts.iloc[min(self.effective_index, len(self.ts) - 1)]
            temp_out_norm = float(row["temp_out_norm"])
            self.temp_out_raw = float(row["temp_amb"])
            solar_irradiance_norm = float(row.get("solar_irradiance_norm", 0.0))
            avg_wind_speed_norm = float(row.get("avg_wind_speed_norm", 0.0))
        else:
            # sim_hour is actual hour of day (0–24); modulo ensures correct
            # wrap-around if the value ever accumulates beyond 24.
            sim_hour = float(states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0]) % 24
            # Synthetic diurnal outdoor temperature profile (normalised)
            if sim_hour < 5:
                temp_out_norm = 0.0
            elif sim_hour < 6:
                temp_out_norm = 0.3
            elif sim_hour < 8:
                temp_out_norm = 0.4
            elif sim_hour < 12:
                temp_out_norm = 0.45
            elif sim_hour < 16:
                temp_out_norm = 0.5
            elif sim_hour < 18:
                temp_out_norm = 0.35
            elif sim_hour < 21.5:
                temp_out_norm = 0.2
            elif sim_hour < 24:
                temp_out_norm = 0.1
            else:
                temp_out_norm = 0.3
            # Compute raw °C from the synthetic normalised value so that
            # get_raw_values() reports a physically consistent temperature.
            self.temp_out_raw = temp_out_norm * self.temp_abs_max

            # NOTE VP 2026.03.10. : Maybe add synthetic data to the other variables as well
            solar_irradiance_norm = 0.0
            avg_wind_speed_norm = 0.0

        # Ensure float32 dtype for all updates
        states["temp_out_norm"][0] = np.float32(temp_out_norm)
        states["solar_irradiance_norm"][0] = np.float32(solar_irradiance_norm)
        states["avg_wind_speed_norm"][0] = np.float32(avg_wind_speed_norm)

        # Expose the temperature scale factor via info so other components
        # (e.g. InsideTemperature) can normalise on the same scale.
        # Not in observation space — raw °C value would destabilise the NN.
        if info is not None:
            info["_temp_abs_max"] = self.temp_abs_max
            info["_wind_speed_abs_max"] = self.wind_speed_abs_max

    def get_raw_values(self) -> dict[str, float]:
        return {"temp_out_raw": self.temp_out_raw}

    def _get_serialize_value(self, param_name: str, value):
        """Handle enum serialization for normalise parameter."""
        if param_name == 'normalise' and isinstance(value, Normalisation):
            return value.value  # Serialize as string
        return super()._get_serialize_value(param_name, value)


# Register WeatherDataSource with the component registry
ComponentRegistry.register('statesource', WeatherDataSource)