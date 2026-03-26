# Data README

## Unified Setup Script (Recommended)

Use `preproc/data_setup.py` to orchestrate all data-fetch and preprocessing
steps from one command.

By default, it runs both pipelines:
- electricity price fetch + preprocessing
- weather/Zenodo fetch + preprocessing

```bash
# Full setup (prices + weather/Zenodo)
python preproc/data_setup.py

# Price-only setup (disable weather pipeline)
python preproc/data_setup.py --skip-weather --years 2023 2024 2025 2026

# Price preprocessing only (reuse local raw files)
python preproc/data_setup.py --skip-weather --years 2025 --skip-price-fetch --raw-price-files data/e_price/awattar/2025_prices.csv

# Price preprocessing with augmentation
python preproc/data_setup.py --skip-weather --years 2023 --skip-price-fetch --raw-price-files data/e_price/e_charts/2023_15m_prices.csv --augment

# Weather-only setup (disable price pipeline)
python preproc/data_setup.py --skip-prices

# Run only specific steps within a pipeline
python preproc/data_setup.py --skip-prices --steps zenodo-extract weather-csv
```

### Typical options

- `--years ...`: target years for price data processing (default: 2017-2026)
- `--price-source awattar|energy-charts`: choose source API
- `--skip-price-fetch`: skip API calls and use local raw CSVs
- `--raw-price-files ...`: explicit local raw CSV inputs
- `--skip-weather`: disable all weather pipelines (WPuQ/Zenodo and DWD)
- `--skip-wpuq`: disable WPuQ/Zenodo weather pipeline only
- `--skip-dwd`: disable DWD weather pipeline only
- `--steps ...`: select which steps to run (price-fetch, price-preproc, zenodo-download, zenodo-extract, weather-csv, sfh-csv; default: all)
- `--augment`: run price augmentation after preprocessing
- `--augment-noise-std`: Gaussian noise std in ct/kWh (default: 0.3)
- `--augment-seed`: random seed for augmentation

Run `python preproc/data_setup.py --help` for full CLI documentation.

### SLURM submission

On HPC clusters, submit data setup as a SLURM job via
`slurm_scripts/slurm_data_setup.sh`. All arguments are forwarded directly to
`preproc/data_setup.py`.

```bash
# Full setup (prices + weather/Zenodo) on a compute node
sbatch slurm_scripts/slurm_data_setup.sh

# Price-only setup
sbatch slurm_scripts/slurm_data_setup.sh --skip-weather --years 2023 2024 2025 2026

# Price preprocessing with augmentation (skip fetch, use local raw files)
sbatch slurm_scripts/slurm_data_setup.sh --skip-weather --skip-price-fetch --years 2023 --raw-price-files data/e_price/awattar/2023_prices.csv --augment

# Weather-only setup
sbatch slurm_scripts/slurm_data_setup.sh --skip-prices --steps zenodo-extract weather-csv sfh-csv
```

SLURM resources: 1 node, 2 CPUs, 10 min wall time (no GPU needed).
Logs are written to `slurm_logs/data_setup/`.

## Electricity Price Data

### Data Source

Electricity prices are fetched from the aWATTar API (German EPEX Spot day-ahead market).
- API docs: https://www.awattar.at/services/api
- Fair-use policy: max 100 requests/day
- Resolution: hourly
- Unit: Eur/MWh
- Region: Germany (DE)

### Preprocessing Pipeline

The pipeline converts raw hourly aWATTar market data into 5-minute resolution
normalized CSV files that the `EnergyPrice` statesource can consume directly.

#### Step 1: Fetch raw prices

Script: `preproc/e_price/awattar_fetch.py`

Fetches hourly EPEX Spot prices for a given year from the aWATTar API and saves
them as CSV. This is the low-level/manual script; for normal usage prefer
`preproc/data_setup.py`.

```bash
python preproc/e_price/awattar_fetch.py
```

Output: `data/e_price/awattar/<YEAR>_prices.csv`
Columns: `start_timestamp, end_timestamp, marketprice, unit, marketprice_eur_per_kwh`

#### Step 2: Preprocess and normalize

Script: `preproc/e_price/awattar_price_preproc.py`

Converts hourly data to 5-minute resolution (forward-fill), converts units from
Eur/MWh to ct/kWh, and applies absolute-max normalization to [-1, 1].

```bash
python preproc/e_price/awattar_price_preproc.py data/e_price/awattar/<YEAR>_prices.csv
# Output: data/e_price/awattar/price_data_<YEAR>_norm.csv

# Or with explicit output path:
python preproc/e_price/awattar_price_preproc.py data/e_price/awattar/<YEAR>_prices.csv -o data/e_price/awattar/custom_output.csv
```

Output: `data/e_price/awattar/price_data_<YEAR>_norm.csv`
Columns: `start, baseprice, unit, hour, price_normalized`

- `baseprice`: price in ct/kWh
- `price_normalized`: absolute-max normalized to [-1, 1], sign-preserving
- `unit`: always `ct/kWh`
- `hour`: hour of day (0-23)

#### Full example (all years)

```bash
# Recommended default full flow (prices + weather/Zenodo):
python preproc/data_setup.py

# Price-only variant:
python preproc/data_setup.py --skip-weather --years 2023 2024 2025 2026

# Equivalent manual flow (aWATTar):
python preproc/e_price/awattar_fetch.py   # repeat per YEAR (edit script constant)
python preproc/e_price/awattar_price_preproc.py data/e_price/awattar/2023_prices.csv
python preproc/e_price/awattar_price_preproc.py data/e_price/awattar/2024_prices.csv
python preproc/e_price/awattar_price_preproc.py data/e_price/awattar/2025_prices.csv
python preproc/e_price/awattar_price_preproc.py data/e_price/awattar/2026_prices.csv
```

### Current Data Files

| File | Description |
|------|-------------|
| `awattar/<YEAR>_prices.csv` | Raw hourly aWATTar fetch output |
| `awattar/price_data_<YEAR>_norm.csv` | Preprocessed 5-min resolution, normalized (aWATTar) |
| `e_charts/<YEAR>_15m_prices.csv` | Raw 15-min Energy Charts fetch output |
| `e_charts/price_data_<YEAR>_norm.csv` | Preprocessed 5-min resolution, normalized (Energy Charts) |
| `prices.csv` | Copy of one year's raw fetch (reference/legacy) |
| `price_data_2025_raw.csv` | Legacy raw price data (SMARD.de, pre-aWATTar) |
| `price_data_2025.xlsx` | Legacy Excel price data (SMARD.de, pre-aWATTar) |

### Visualization

Use `preproc/e_price/plot_price.ipynb` to inspect and plot the preprocessed price data.

---

## Weather data

### Zenodo: https://zenodo.org/records/5642902

Paper: Dataset on electrical single-family house and heat pump load profiles in Germany

Link: https://www.nature.com/articles/s41597-022-01156-1#Tab3

"Weather data
We obtain weather data such as outdoor air temperature, wind speed, relative humidity and solar global radiation from the weather service wetter-online17 for the location Hamelin. We request the current weather data in intervals of 5 min from the weather service via a HTTP REST API and store it on the database server. The year 2018 has irregular intervals of 1 min to 1 hour."

Weather data in WPuQ: https://wo.wetteronline.de/

API docs wetteronline: https://wetteronline.readthedocs.io/en/latest/

--> WetterOnline -- rather leave it

### Zenodo/WPuQ Preprocessing Pipeline

Converts HDF5 archives from Zenodo into CSV files for the environment statesources.

#### Step 1: Download and extract Zenodo archives

Downloads files listed in `data/weather/zenodo/ds_links.txt`, extracts zips
into HDF5 files (`*_weather.hdf5`, `*_data_1min.hdf5`).

#### Step 2: Extract weather CSV from HDF5

Script: `preproc/weather/extract_weather_csv.py`

Reads `WEATHER_SERVICE/IN` from the HDF5 file, merges all variables (inner
join), drops NaN rows, renames columns, and drops unused ones.

Column renames:

| HDF5 variable | Renamed to | Meaning |
|---------------|------------|---------|
| `temperature` | `temp_amb` | Ambient temperature |
| `relative_humidity` | `rel_humidity` | Relative humidity |
| `solar_irradiance` | `direct_sun_shine` | Solar global radiation |
| `wind_direction` | `wind_dir` | Wind direction |
| `wind_speed` | `avg_wind_speed` | Wind speed |

Dropped columns: `atmospheric_pressure`, `precipitation_rate`,
`probability_of_precipitation`, `apparent_temperature`, `wind_gust_speed`

```bash
python preproc/weather/extract_weather_csv.py --input data/weather/zenodo/2018_weather.hdf5
# Output: data/weather/zenodo/csvs_weather/2018_weather.csv
```

#### Step 3: Extract SFH (Single Family Home) CSV from HDF5

Script: `preproc/weather/extract_sfh_csv.py`

Reads `NO_PV` group: per building (SFH10, SFH11, …), keeps `_TOT` columns from
HEATPUMP (`hp_` prefix) and HOUSEHOLD (`hh_` prefix), merges on timestamp.

```bash
python preproc/weather/extract_sfh_csv.py --input data/weather/zenodo/2018_data_1min.hdf5
# Output: data/weather/zenodo/csvs_2018_data_1min/SFH10.csv, SFH11.csv, ...
```

#### Full example (Zenodo/WPuQ)

```bash
python preproc/data_setup.py --skip-prices
# Or skip download if HDF5 files exist:
python preproc/data_setup.py --skip-prices --steps zenodo-extract weather-csv sfh-csv
```

Use `preproc/weather/explore_hdf5.py` or `explore_hdf5_notebook.ipynb` to
inspect HDF5 structure before extraction.

---

### DWD (Deutscher Wetterdienst)

DWD CDC open-data server, 10-minute resolution.
Link: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/
Station map: https://www.dwd.de/DE/fachnutzer/landwirtschaft/appl/stationskarte/_node.html
Default station: **04177** (Rheinstetten). Data types: `wind`, `solar`, `air_temperature`.

#### Step 1: Fetch raw data

Script: `preproc/weather/dwd/dwd_fetch.py`

Downloads historical + recent zip archives, extracts `produkt_*.txt` files,
converts to CSV, concatenates and deduplicates per data type.

#### Step 2: Preprocess and merge

Script: `preproc/weather/dwd/dwd_preprocess.py`

Selects relevant columns, merges on `MESS_DATUM` (outer join), renames, adds
combined solar column, converts timestamp to UTC datetime, drops all-missing
(`-999`) rows. Writes full merged CSV at 10-minute resolution.

Column renames:

| DWD raw | Renamed to | Meaning |
|---------|------------|---------|
| `MESS_DATUM` | `timestamp` | Measurement date/time |
| `FF_10` | `avg_wind_speed` | Wind speed (m/s) |
| `DD_10` | `wind_dir` | Wind direction (°) |
| `GS_10` | `direct_sun_shine` | Global solar irradiance (J/cm²) |
| `DS_10` | `diff_sun_shine` | Diffuse solar irradiance (J/cm²) |
| `TT_10` | `temp_amb` | Ambient temperature (°C) |
| `RF_10` | `rel_humidity` | Relative humidity (%) |

Derived column: `sun_shine = direct_sun_shine + diff_sun_shine`

#### Step 3: Upsample to 5-minute resolution

Per-year upsample from 10-min to 5-min:
- `average` (default): linear interpolation, skips `-999` values
- `duplicate`: forward-fill

#### Step 4 (optional): Normalize and augment

- `--normalize`: absolute-max normalization to [-1, 1], `-999` → NaN
- `--augment` (via `data_setup.py`): per-column Gaussian noise, configured in
  `preproc/augment_config.yaml`

Per-year `<YEAR>_missing_entries.txt` reports list days with `-999` values.

#### Full example (DWD)

```bash
python preproc/data_setup.py --skip-prices --skip-wpuq --steps dwd-fetch dwd-preprocess
# With normalization:
python preproc/data_setup.py --skip-prices --skip-wpuq --steps dwd-fetch dwd-preprocess --normalize
# Standalone:
python preproc/weather/dwd/dwd_preprocess.py --normalize
```

### Current Weather Data Files

| File / Directory | Description |
|------------------|-------------|
| `weather/zenodo/*_weather.hdf5` | Raw HDF5 weather archives from Zenodo |
| `weather/zenodo/*_data_1min.hdf5` | Raw HDF5 SFH load profiles (1-min) |
| `weather/zenodo/csvs_weather/*.csv` | Extracted weather CSVs (5-min) |
| `weather/zenodo/csvs_<stem>/*.csv` | Extracted SFH CSVs (one per building) |
| `weather/dwd/preprocessed/merged_04177.csv` | Full merged DWD data (10-min) |
| `weather/dwd/preprocessed/<YEAR>_merged_04177.csv` | Per-year DWD (5-min) |
| `weather/dwd/preprocessed/<YEAR>_merged_04177_norm.csv` | Per-year DWD (5-min, normalized) |

### Weather pipeline via unified script

```bash
python preproc/data_setup.py --skip-prices                    # Full weather setup
python preproc/data_setup.py --skip-prices --skip-dwd         # Zenodo only
python preproc/data_setup.py --skip-prices --skip-wpuq        # DWD only
```

## Other Data Sources (Reference)

SMARD.de (legacy source, replaced by aWATTar):
Source link: https://www.smard.de/en/downloadcenter/download-market-data/?downloadAttributes=%7B%22selectedCategory%22:3,%22selectedSubCategory%22:8,%22selectedRegion%22:%22DE%22,%22selectedFileType%22:%22CSV%22,%22from%22:1770591600000,%22to%22:1772060399999%7D

Global energy price dataset:
Link: https://zenodo.org/records/16284828
Take a look, but resolution may not be okay

Solar and wind data:
https://www.renewables.ninja/
Only hourly -- interpolation in between (?)

API for day ahead energy prices, 15min resolution:
https://api.energy-charts.info/#/prices/day_ahead_price_price_get
--> Scripts are fetching these data as well

Global day ahead electricity price dataset:
Link: https://ieee-dataport.org/documents/global-day-ahead-electricity-price-dataset
Link: https://data.mendeley.com/datasets/s54n4tyyz4/1
Processing and data collection method: https://github.com/d3m-lab/Energy-Price-Data-Research

Tado energy API, aWATTar:
https://energy.tado.com/services/api
--> Current implementation uses TADO / aWATTar API -- free and easy to use.

Tibber API, needs GraphQL:
https://developer.tibber.com/explorer
https://developer.tibber.com/docs/reference
--> Issue: needs tibber account, which needs a contract with tibber as well -- out of scope

Transparency platform API:
https://documenter.getpostman.com/view/7009892/2s93JtP3F6#3b383df0-ada2-49fe-9a50-98b1bb201c6b

Publication similar to my thesis:
https://research.wu.ac.at/de/publications/deep-learning-in-energy-modeling-application-in-smart-buildings-w/

TODO VP: Maybe integrate FAIR-RS: https://www.rdm.kit.edu/english/servicetools_tools_fair-rs.php

TODO VP look up radar: https://www.radar-service.eu/radar/en/home

TODO VP: maybe fetch additional weather data from non-EU hourly datasources -- EnergyPlus, link: https://energyplus.net/weather/sources