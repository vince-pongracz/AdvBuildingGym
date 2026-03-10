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
- `--skip-weather`: disable weather/Zenodo pipeline
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

### DWD



Station map: https://www.dwd.de/DE/fachnutzer/landwirtschaft/appl/stationskarte/_node.html

### Weather/Zenodo pipeline via unified script

```bash
# Download Zenodo files listed in data/weather/zenodo/ds_links.txt,
# extract zip archives, and run weather+SFH CSV extraction
python preproc/data_setup.py

# If files are already present locally, skip download and run weather only:
python preproc/data_setup.py --skip-prices --steps zenodo-extract weather-csv sfh-csv
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

TODO VP
Maybe integrate FAIR-RS: https://www.rdm.kit.edu/english/servicetools_tools_fair-rs.php

TODO VP look up radar: https://www.radar-service.eu/radar/en/home
