import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..csv_lookahead import CsvLookahead
from ..reloadable import CsvReloadable
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import TEMP_ABS_MAX_CELSIUS
from adv_building_gym._common.normalisation import Normalisation, normalise_with_scale_factor

logger = logging.getLogger(__name__)


class WeatherDataSource(StateSource, CsvLookahead, CsvReloadable):
    """Ambient temperature, wind speed, and solar irradiance from a preprocessed weather CSV.

    Units (raw, before runtime normalisation):
        - ``temp_amb`` / ``raw_temp_out``         : °C
        - ``avg_wind_speed`` / ``raw_wind_speed`` : m/s
        - ``sun_shine`` / ``raw_solar_irradiance``: W/m² (mean over the 5-min step)

    Both DWD and Zenodo/WPuQ pipelines emit ``sun_shine`` in W/m² (DWD converted from
    J/cm² per 10 min in ``dwd_preprocess.py``; Zenodo native). ``ctxt_solar_irradiance_max``
    is the scale factor recovering raw W/m² from ``s_solar_irradiance_norm ∈ [0, 1]``.
    """

    # normalise is an enum, need special handling for serialization
    _context_params: ClassVar[Set[str]] = {'control_step'}
    _exclude_params: ClassVar[Set[str]] = {
        'temp_out_raw',
        'wind_speed_abs_max', 'wind_speed_raw',
        'sun_shine_abs_max', 'sun_shine_raw',
    }

    # Lookahead channels feeding the generators' power forecasts (logical key -> CSV column).
    _lookahead_columns: ClassVar[dict[str, str]] = {
        "solar_W_m2": "sun_shine",
        "wind_ms": "avg_wind_speed",
    }

    def __init__(self, name: str, ds_path: str | None = None,
                normalise: Normalisation | str | None = Normalisation.MAX_ABS_SCALING) -> None:
        super().__init__(name=name)

        self.normalise = Normalisation.init(normalise)  # Store for serialization
        # Raw values for get_raw_values() — updated each step
        self.temp_out_raw: float = 0.0
        self.wind_speed_raw: float = 0.0
        # Solar irradiance (W/m², 5-min mean).
        self.sun_shine_raw: float = 0.0
        self.sun_shine_abs_max: float = 0.0

        # Owned look-ahead channel (logical key -> future raw values), updated in place and
        # republished on info["raw_weather_forecast"] each step. Generators hold a reference to
        # THIS dict only and still see the row[t+1] look-ahead written after their own update_state.
        self._weather_forecast: dict[str, list[float]] = {}

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

        # Temperature uses a FIXED scale (TEMP_ABS_MAX_CELSIUS, not the per-variant max) so
        # s_temp_out_norm is identical across weather variants. 
        # Irradiance and wind are data-driven 
        # (scale factors exposed separately to reconstruct raw W/m² and m/s).
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

        # Each step reads raw temp_amb / sun_shine / avg_wind_speed (the latter two also feed the
        # generators' lookahead) plus the fixed-scale s_temp_out_norm. The irradiance/wind norm
        # columns are unused, so drop them with the rest of the CSV.
        self._keep_ts_columns({"temp_amb", "sun_shine", "avg_wind_speed", "s_temp_out_norm"})

    # The temperature scale is a fixed constant (TEMP_ABS_MAX_CELSIUS), not a per-variant
    # factor, so it is not published as a ctxt_* observation but handed to consumers via
    # info["temp_abs_max"] (see update_state). The weather drivers (outdoor temp, irradiance,
    # wind) are not observations either: the policy sees the *effects* (s_temp_error_norm,
    # s_pv_power_norm, s_wind_power_norm).

    def update_state(self, states, info: dict) -> None:
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

        # Hand the weather drivers to consumers via the shared info channel — none are policy
        # observations. Outdoor temp (norm) → BuildingHeatLoss physics; raw irradiance (W/m²)
        # and wind speed (m/s) → the generators' power curves.
        # Fixed temp scale (°C) for every temp consumer. Constant; single publication point.
        info["temp_abs_max"] = TEMP_ABS_MAX_CELSIUS
        info["temp_out_norm"] = temp_out_norm
        info["raw_solar_irradiance_W_m2"] = self.sun_shine_raw
        info["raw_wind_speed_ms"] = self.wind_speed_raw

        # When forecasting is active, pre-compute future RAW weather (W/m², m/s) over the
        # canonical step set and publish it as a plain channel, so generators read
        # info["raw_weather_forecast"][<channel>].
        steps = info.get("forecast_steps")
        if steps:
            self._weather_forecast.clear()
            self._weather_forecast.update(self.lookahead(list(steps)))
            info["raw_weather_forecast"] = self._weather_forecast

    # Weather drivers are not policy observations. 
    # Generators consume future irradiance/wind from info["raw_weather_forecast"]

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