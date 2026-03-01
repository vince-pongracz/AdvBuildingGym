# Data README

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

Script: `preproc/awattar_fetch.py`

Fetches hourly EPEX Spot prices for a given year from the aWATTar API and saves
them as CSV. Change the `YEAR` constant in the script to select the target year.

```bash
python preproc/awattar_fetch.py
```

Output: `data/<YEAR>_prices.csv`
Columns: `start_timestamp, end_timestamp, marketprice, unit, marketprice_eur_per_kwh`

#### Step 2: Preprocess and normalize

Script: `preproc/awattar_price_preproc.py`

Converts hourly data to 5-minute resolution (forward-fill), converts units from
Eur/MWh to ct/kWh, and applies absolute-max normalization to [-1, 1].

```bash
python preproc/awattar_price_preproc.py data/<YEAR>_prices.csv
# Output: data/price_data_<YEAR>_norm.csv

# Or with explicit output path:
python preproc/awattar_price_preproc.py data/<YEAR>_prices.csv -o data/custom_output.csv
```

Output: `data/price_data_<YEAR>_norm.csv`
Columns: `start, baseprice, unit, hour, price_normalized`

- `baseprice`: price in ct/kWh
- `price_normalized`: absolute-max normalized to [-1, 1], sign-preserving
- `unit`: always `ct/kWh`
- `hour`: hour of day (0-23)

#### Full example (all years)

```bash
# Edit YEAR in awattar_fetch.py for each year, then run:
python preproc/awattar_fetch.py   # repeat for 2023, 2024, 2025, 2026

# Preprocess all fetched files:
python preproc/awattar_price_preproc.py data/2023_prices.csv
python preproc/awattar_price_preproc.py data/2024_prices.csv
python preproc/awattar_price_preproc.py data/2025_prices.csv
python preproc/awattar_price_preproc.py data/2026_prices.csv
```

### Current Data Files

| File | Description |
|------|-------------|
| `<YEAR>_prices.csv` | Raw hourly aWATTar fetch output (2023-2026) |
| `price_data_<YEAR>_norm.csv` | Preprocessed 5-min resolution, normalized (2023-2026) |
| `prices.csv` | Copy of one year's raw fetch (reference/legacy) |
| `price_data_2025_raw.csv` | Legacy raw price data (SMARD.de, pre-aWATTar) |
| `price_data_2025.xlsx` | Legacy Excel price data (SMARD.de, pre-aWATTar) |

### Visualization

Use `preproc/plot_price.ipynb` to inspect and plot the preprocessed price data.

---

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
