import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..csv_lookahead import CsvLookahead
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import TEMP_ABS_MAX_CELSIUS
from adv_building_gym._common.normalisation import Normalisation, normalise_with_scale_factor

logger = logging.getLogger(__name__)


class WeatherDataSource(StateSource, CsvLookahead):
    """Ambient temperature, wind speed, and solar irradiance from a preprocessed weather CSV.

    Units (raw, before runtime normalisation):
        - ``temp_amb`` / ``raw_temp_out``         : °C
        - ``avg_wind_speed`` / ``raw_wind_speed`` : m/s
        - ``sun_shine`` / ``raw_solar_irradiance``: W/m² (mean over the 5-min step)

    Both DWD and Zenodo/WPuQ preprocessing pipelines emit ``sun_shine`` in W/m²
    (DWD is converted from J/cm² per 10 min in ``dwd_preprocess.py``;
    Zenodo is native W/m²). The ``ctxt_solar_irradiance_max`` context entry
    is the scale factor used to recover the raw W/m² from the normalised
    ``s_solar_irradiance_norm ∈ [0, 1]`` observation.
    """

    # normalise is an enum, need special handling for serialization
    _context_params: ClassVar[Set[str]] = {'control_step'}
    _exclude_params: ClassVar[Set[str]] = {
        'temp_out_raw',
        'wind_speed_abs_max', 'wind_speed_raw',
        'sun_shine_abs_max', 'sun_shine_raw',
    }

    def __init__(self, name: str, ds_path: str | None = None,
                normalise: Normalisation | str | None = Normalisation.MAX_ABS_SCALING) -> None:
        super().__init__(name=name)

        self.normalise = Normalisation.init(normalise)  # Store for serialization
        # Raw values for get_raw_values() — updated each step
        self.temp_out_raw: float = 0.0
        self.wind_speed_raw: float = 0.0
        # Solar irradiance (W/m², 5-min mean); both DWD and Zenodo/WPuQ CSVs deliver W/m².
        self.sun_shine_raw: float = 0.0
        self.sun_shine_abs_max: float = 0.0

        # loader auto-fires _run_post_load after each read; attrs it needs MUST be set above.
        self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)
        if ds_path is not None:
            logger.info("Use data file: %s", ds_path)
        else:
            logger.debug("No initial data file for '%s', data source will be assigned by DataCombinator", name)

    def _post_load_data_processing(self) -> None:
        """Normalise weather columns after load/reload (cleaning is done in preprocessing)."""
        # warn only on a new file (not every same-file reload at episode reset)
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

        # Temperature uses a FIXED normalisation scale (TEMP_ABS_MAX_CELSIUS, not the
        # per-variant data maximum) so s_temp_out_norm has an identical scale across every
        # weather variant. Irradiance and wind stay data-driven (their scale factors are
        # exposed separately and used to reconstruct raw W/m² and m/s).
        if "temp_amb" in self.ts.columns:
            self.ts["s_temp_out_norm"] = (self.ts["temp_amb"] / TEMP_ABS_MAX_CELSIUS).astype(np.float32)

        # Normalise raw columns + derive scale factors for raw↔norm conversion.
        # cols = { raw_col: (norm_col, scale_attr) }
        cols = {
            "sun_shine": ("s_solar_irradiance_norm", "sun_shine_abs_max"),
            "avg_wind_speed": ("s_avg_wind_speed_norm", "wind_speed_abs_max"),
        }

        for raw_col, (norm_col, scale_attr) in cols.items():
            if raw_col in self.ts.columns:
                normalised, scale_factor = normalise_with_scale_factor(self.ts[raw_col], self.normalise)
                self.ts[norm_col] = normalised
                if scale_attr is not None:
                    setattr(self, scale_attr, scale_factor)

        # Day-of-year normalised to [0, 1] (divisor = year length; leap → 366)
        if "timestamp" in self.ts.columns:
            ts_parsed = pd.to_datetime(self.ts["timestamp"], utc=True, errors="coerce")
            year_length = np.where(ts_parsed.dt.is_leap_year, 366.0, 365.0)
            self.ts["s_date"] = ((ts_parsed.dt.dayofyear - 1) / year_length).astype(np.float32)
        else:
            if self.is_new_data_source:
                logger.warning("WeatherDataSource '%s': no 'timestamp' column — s_date set to 0.", self.name)
            self.ts["s_date"] = np.float32(0.0)


    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict
                    ) -> tuple[OrderedDict, OrderedDict]:

        # Day-of-year of the current row, [0, 1]; recomputed per step for midnight crossings.
        if "s_date" not in state_spaces.keys():
            state_spaces["s_date"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # The temperature normalisation scale is a fixed constant (TEMP_ABS_MAX_CELSIUS),
        # not a data-driven per-variant factor, so it is no longer published as a ctxt_*
        # observation. It is handed to the temperature consumers on the info channel
        # (info["temp_abs_max"], see update_state). Weather drivers themselves (outdoor
        # temperature, irradiance, wind speed) are NOT observations either: the policy sees
        # the *effects* instead (s_temp_error_norm, s_pv_power_norm, s_wind_power_norm).

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
        self.wind_speed_raw = float(row.get("avg_wind_speed", 0.0))
        self.sun_shine_raw = float(row.get("sun_shine", 0.0))

        states["s_date"][0] = np.float32(row.get("s_date", 0.0))

        # Hand the weather drivers to their consumers via the shared info channel — none
        # of them are policy observations. Outdoor temperature (normalised) → BuildingHeatLoss
        # physics; raw irradiance (W/m²) and wind speed (m/s) → the generators' power curves.
        # A self-reference lets the generators query the future-weather look-ahead in forecast().
        # Read at the next step, mirroring the previous obs-buffer one-step structure.
        if info is not None:
            # Fixed temperature normalisation scale (°C) for every temperature consumer
            # (HP / BuildingHeatLoss / InsideTemperature / temp rewards). Constant, not
            # data-driven; this is the single publication point on the shared channel.
            info["temp_abs_max"] = TEMP_ABS_MAX_CELSIUS
            info["temp_out_norm"] = temp_out_norm
            info["raw_solar_irradiance_W_m2"] = self.sun_shine_raw
            info["raw_wind_speed_ms"] = self.wind_speed_raw
            info["_weather_source"] = self

    # No weather driver is a policy observation, so the weather source publishes no
    # s_fc_* keys and is not Forecastable. It still feeds the generators' power forecasts
    # via raw_weather_forecast() below (see SolarPanel / WindTurbine.forecast), using the
    # CsvLookahead mixin's cached future-row reads.
    def raw_weather_forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        """Future raw irradiance (W/m²) and wind speed (m/s) for the generator power
        forecasts — values at ``effective_index + step`` for each requested offset."""
        return {
            "solar_W_m2": self._csv_forecast(self.ts, self.effective_index, "sun_shine", selected_future_steps),
            "wind_ms": self._csv_forecast(self.ts, self.effective_index, "avg_wind_speed", selected_future_steps),
        }

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


# register with ComponentRegistry
ComponentRegistry.register('statesource', WeatherDataSource)