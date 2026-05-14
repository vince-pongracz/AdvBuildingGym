# Data Sources and Physical Units — Complete Audit

## 1. Weather data

### 1a. DWD (Deutscher Wetterdienst), station 04177 (Rheinstetten)

Source: DWD CDC open-data server, 10-min raw → upsampled to 5-min.
Fetch/preprocess: `preprocessing/weather/dwd/` → `data/weather/dwd/preprocessed/YYYY_merged_04177.csv`

Columns in the preprocessed CSV (note: irradiance is already converted from raw DWD J/cm² per 10 min to W/m² in `dwd_preprocess.py:83-89`):

| Column | DWD raw key | Unit | Description |
| --- | --- | --- | --- |
| `timestamp` | `MESS_DATUM` | UTC datetime | Measurement time |
| `avg_wind_speed` | `FF_10` | m/s | 10-min mean wind speed |
| `wind_dir` | `DD_10` | degrees (°) | 10-min mean wind direction |
| `direct_sun_shine` | `GS_10` | W/m² | Global shortwave irradiance (converted from J/cm²/10min) |
| `diff_sun_shine` | `DS_10` | W/m² | Diffuse shortwave irradiance (converted from J/cm²/10min) |
| `sun_shine` | derived | W/m² | `direct_sun_shine + diff_sun_shine` |
| `temp_amb` | `TT_10` | °C | Ambient air temperature at 2 m |
| `rel_humidity` | `RF_10` | % (0–100) | Relative humidity |

Conversion factor used: J/cm² per 10 min × 10000/600 = × 50/3 ≈ 16.667 → W/m².

### 1b. Zenodo / WPuQ weather

Source: Zenodo 5642902 (Schlemminger et al., *Scientific Data* 2022), WetterOnline service, location Hamelin. Raw cadence varies per year — some files ship at 5-min, others at hourly/irregular intervals; the preprocessing step linearly interpolates everything onto a uniform 5-min grid.
Preprocess: `preprocessing/weather/extract_weather_csv.py` → `data/weather/zenodo/csvs_weather/`

| HDF5 var | Renamed | Unit |
| --- | --- | --- |
| `temperature` | `temp_amb` | °C |
| `relative_humidity` | `rel_humidity` | % |
| `solar_irradiance` | `direct_sun_shine` (+ `sun_shine` alias) | W/m² (native) |
| `wind_direction` | `wind_dir` | ° |
| `wind_speed` | `avg_wind_speed` | m/s |

Both DWD and Zenodo pipelines emit `sun_shine` in W/m² — see `weather.py:23-29`.

## 2. Electricity price

Sources:

- aWATTar / EPEX Spot day-ahead — raw fetched in EUR/MWh, hourly.
- Energy-Charts API — raw fetched in EUR/MWh, hourly or 15-min depending on bidding zone and date range (interval is inferred from the response; see [`energy_charts_fetch.py:145-149`](../preprocessing/e_price/energy_charts_fetch.py#L145-L149)).

Preprocess: `preprocessing/e_price/awattar_price_preproc.py` — converts to ct/kWh and resamples to 5-min (forward-fill).
Output: `data/e_price/awattar/price_data_YYYY.csv`, `data/e_price/e_charts/price_data_YYYY.csv`

| Column | Unit | Description |
| --- | --- | --- |
| `start` | datetime | Step start timestamp (5-min) |
| `baseprice` | ct/kWh | Day-ahead spot price |
| `unit` | string label | Always literal `ct/kWh` |
| `hour` | 0–23 int | Hour of day |

Raw-fetch CSV (`<YEAR>_prices.csv`) has `marketprice` in EUR/MWh and `marketprice_ct_per_kwh` in ct/kWh (1 EUR/MWh = 0.1 ct/kWh).

## 3. Household consumption (WPuQ SFH profiles)

Source: Zenodo WPuQ `*_data_1min.hdf5` `NO_PV` group, 1-min raw, W.
Preprocess: `preprocessing/hh_consumption/extract_hh_consumption.py` — resamples 1-min → 5-min, converts W → kW.
Output: `data/hh_consumption/wpuq/YYYY_SFH<id>.csv`

| Column | Unit | Description |
| --- | --- | --- |
| `timestamp` | datetime | Sample timestamp (5-min) |
| `hh_consumption_kW` | kW | Household electrical load (mean over 5-min) |

Aggregated variant `YYYY_aggregated_consumption.csv` carries the same `hh_consumption_kW` column. Consumed by `DesiredUserEnergyNeed`.

## 4. Desired inside temperature

Source: hand-authored setpoint profiles.
Files: `data/inside_temp/inside_temp_*.csv`

| Column | Unit | Description |
| --- | --- | --- |
| `timestamp` | datetime (5-min) | Step time |
| `desired_temp_in [°C]` | °C | Setpoint for indoor air |

Read by `InsideTemperature`; fallback alias accepted: `desired_temp_in` (also °C).

## 5. EV usage profile (schedule of connect/disconnect events)

Files: `data/ev_usage_profiles/ev_*.csv` — sparse rows; populated row = CONNECT event, blank row = DISCONNECT event. Parsed by `EVState`.

| Column | Unit | Description |
| --- | --- | --- |
| `start` | datetime | Event time |
| `max_cap_kWh` | kWh | EV battery capacity |
| `max_charging_kW` | kW | Max charge/discharge power at the connector |
| `charger_efficiency` | dimensionless 0–1 | Charging efficiency (η_in) |
| `discharge_efficiency` | dimensionless 0–1 | Discharging efficiency (η_out, V2G) |
| `v2g_enabled` | bool | Whether discharge to grid is allowed |
| `start_soc` | fraction 0–1 | SoC at connect time |
| `target_soc` | fraction 0–1 | SoC target by deadline |
| `target_soc_reach_duration_h` | hours | Deadline duration after start |

## 6. Operator energy control (grid limit)

No CSV — set as a constant `max_power_kW` in the env config YAML and published every step. Unit: kW. (`operator_energy_control.py`)

## Summary — runtime keys vs. raw units

### Policy-visible observations (`s_*`, `raw_sim_hour`)

| Key | Unit at exposure |
| --- | --- |
| `s_temp_out_norm` | normalised, scale = `ctxt_temp_abs_max` (°C) |
| `s_solar_irradiance_norm` | normalised [0,1], scale = `ctxt_solar_irradiance_max` (W/m²) |
| `s_avg_wind_speed_norm` | normalised [0,1], scale = `ctxt_wind_speed_abs_max` (m/s) |
| `s_E_price` | normalised [-1,1], scale = `ctxt_E_price_max` (ct/kWh) |
| `s_temp_in_norm` | normalised (÷ `ctxt_temp_abs_max`, °C) |
| `raw_sim_hour` | hours (0–24) |

### Context / scale factors (`ctxt_*`, static within an episode)

| Key | Unit |
| --- | --- |
| `ctxt_temp_abs_max` | °C |
| `ctxt_solar_irradiance_max` | W/m² |
| `ctxt_wind_speed_abs_max` | m/s |
| `ctxt_E_price_max` | ct/kWh |
| `ctxt_operator_max_power_kW` | kW |
| `ctxt_battery_capacity_kWh` | kWh |
| `ctxt_hp_max_power_kW` | kW (electrical); thermal via × `cop_heat` / `cop_cool` |

### Diagnostic / shared info dict (`info[...]`, not in obs space)

| Key | Unit |
| --- | --- |
| `raw_temp_out` | °C |
| `raw_wind_speed` | m/s |
| `raw_solar_irradiance` | W/m² |
| `raw_E_price` | ct/kWh |
| `raw_desired_temp_in` | °C |
| `net_power_kW`, `power_breakdown[*]` | kW (signed: + consume / − export) |
| `cum_E_kWh` | kWh |
| `ev_schedule_max_cap_kWh` | kWh |
| `ev_schedule_max_charging_kW` | kW |

Building physical params (in `BuildingProps`): `K` in W/K (heat transfer), `mC` in J/K (thermal capacitance), `control_step` in seconds.
