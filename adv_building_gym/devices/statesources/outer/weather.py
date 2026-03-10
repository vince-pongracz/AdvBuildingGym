import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.config.utils.serializable import ComponentRegistry
from adv_building_gym.utils.normalisation import Normalisation, normalise_series

logger = logging.getLogger(__name__)


class WeatherDataSource(StateSource):
    """WeatherDataSource"""

    # normalise is an enum, need special handling for serialization
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'ts'}

    def __init__(self, name: str, ds_path: str | None = None,
                 normalise: Normalisation | str | None = Normalisation.MAX_ABS_SCALING) -> None:
        super().__init__(name, ds_path)

        self.normalise = Normalisation.init(normalise)  # Store for serialization

        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            self._post_load_data_processing()
        else:
            logger.info("No data file provided, will use synthetic data")

    def _post_load_data_processing(self) -> None:
        """Normalise weather columns after CSV load / reload."""
        # Zenodo CSVs have direct_sun_shine only (global irradiance); create sun_shine alias
        if "sun_shine" not in self.ts.columns and "direct_sun_shine" in self.ts.columns:
            self.ts["sun_shine"] = self.ts["direct_sun_shine"]

        cols = {
            "temp_amb": "temp_out_norm",
            "sun_shine": "solar_irradiance_norm",
            "avg_wind_speed": "avg_wind_speed_norm",
        }

        for raw_col, norm_col in cols.items():
            if raw_col in self.ts.columns:
                self.ts[norm_col] = normalise_series(self.ts[raw_col], self.normalise)


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

    def update_state(self, states) -> None:
        if self.ts is not None:
            row = self.ts.iloc[min(self.effective_index, len(self.ts) - 1)]
            temp_out_norm = float(row["temp_out_norm"])
            solar_irradiance_norm = float(row.get("solar_irradiance_norm", 0.0))
            avg_wind_speed_norm = float(row.get("avg_wind_speed_norm", 0.0))
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0]
            # Apply a simple time-based temperature profile if no CSV data is provided
            if current_sim_hour < 5:
                temp_out_norm = 0.0
            elif current_sim_hour < 6:
                temp_out_norm = 0.3
            elif current_sim_hour < 8:
                temp_out_norm = 0.4
            elif current_sim_hour < 12:
                temp_out_norm = 0.45
            elif current_sim_hour < 16:
                temp_out_norm = 0.5
            elif current_sim_hour < 18:
                temp_out_norm = 0.35
            elif current_sim_hour < 21.5:
                temp_out_norm = 0.2
            elif current_sim_hour < 24:
                temp_out_norm = 0.1
            else:
                temp_out_norm = 0.3
            # NOTE VP 2026.03.10. : Maybe add synthetic data to the other variables as well
            solar_irradiance_norm = 0.0
            avg_wind_speed_norm = 0.0

        # Ensure float32 dtype for all updates
        states["temp_out_norm"][0] = np.float32(temp_out_norm)
        states["solar_irradiance_norm"][0] = np.float32(solar_irradiance_norm)
        states["avg_wind_speed_norm"][0] = np.float32(avg_wind_speed_norm)

    def _get_serialize_value(self, param_name: str, value):
        """Handle enum serialization for normalise parameter."""
        if param_name == 'normalise' and isinstance(value, Normalisation):
            return value.value  # Serialize as string
        return super()._get_serialize_value(param_name, value)


# Register WeatherDataSource with the component registry
ComponentRegistry.register('statesource', WeatherDataSource)