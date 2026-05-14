# Synthetic Dataset Generator

Generates noised / shifted copies of preprocessed price and weather CSVs to
expand the scenario pool used by `DataCombinator`.

## Layout

```
preprocessing/
├── synthesize.py             # transform engine + CLI + pipeline entry point
├── synthesize_config.yaml    # top-level: which syn_cfgs to apply
└── syn_cfgs/
    ├── syn_cfg_0.yaml            # noise-only baseline (no constant shift)
    ├── syn_cfg_1_neg.yaml        # weak negative shifts (cooler / cheaper)
    ├── syn_cfg_1_pos.yaml        # weak positive shifts (warmer / pricier)
    ├── syn_cfg_2_neg.yaml        # medium negative shifts
    ├── syn_cfg_2_pos.yaml        # medium positive shifts
    ├── syn_cfg_3_neg.yaml        # strong negative shifts
    └── syn_cfg_3_pos.yaml        # strong positive shifts
```

## How it works

1. The unified pipeline ([data_setup.py](data_setup.py)) runs price / WPuQ /
   DWD / hh-consumption preprocessing as before.
2. If `--synthesize` is passed, `run_synthesize` (in [pipelines.py](pipelines.py))
   reads `synthesize_config.yaml`, loads each listed `syn_cfg_*.yaml`, and for
   every preprocessed CSV writes a sibling `<stem>_<syn_cfg_name>.csv` next to
   the source file.
3. `discover_synthetic_scenarios` (in
   [adv_building_gym/config/utils/discover_scenarios.py](../adv_building_gym/config/utils/discover_scenarios.py))
   globs for `*_syn_cfg_*.csv` so the generated files are picked up
   automatically when `include_synthesized: true` is set in the data combinator
   config.

## Top-level config (`synthesize_config.yaml`)

```yaml
seed: 42                             # base seed; per-(cfg, file) seeds are derived
syn_cfg_dir: preprocessing/syn_cfgs  # where syn_cfg_*.yaml-s live
active_configs: [syn_cfg_1_neg, syn_cfg_1_pos, syn_cfg_2_neg, syn_cfg_2_pos, syn_cfg_3_neg, syn_cfg_3_pos]
```

## Per-level config (`syn_cfg_*.yaml`)

Each level defines, under `price` and `weather`, a per-column transform
pipeline + optional clip bounds:

```yaml
name: syn_cfg_2_pos

price:
  baseprice:
    transforms:
      - {type: gaussian_noise, std: 0.3, smooth: {kind: moving_average, window: 6}}
      - {type: constant_shift, value: +1.0}

weather:
  temp_amb:
    transforms:
      - {type: gaussian_noise, std: 0.5, smooth: {kind: moving_average, window: 6}}
      - {type: constant_shift, value: +0.5}
    clip: {min: -50.0, max: 60.0}
```

Transforms apply in order. Clipping is always the final step.

The preset configs intentionally share noise levels and only differ in
their `constant_shift` values. They come in `_pos` / `_neg` pairs (same
magnitude, opposite sign) at three intensities (weak / medium / strong) plus
a noise-only baseline (`syn_cfg_0`), so the synthesised datasets cover a
symmetric grid of "climate / market" offsets at the same noise budget.

## Available transforms

| `type`           | params                                                         | notes                                                  |
|------------------|----------------------------------------------------------------|--------------------------------------------------------|
| `gaussian_noise` | `std`, `smooth: {kind, ...}` (optional)                        | additive noise; smoothing is rescaled back to `std`    |
| `constant_shift` | `value`                                                        | adds a constant to the whole series                    |
| `linscaler`      | `factor`                                                       | uniform multiplicative gain: `value * factor`          |

### Smoothing kinds (avoids high-frequency artifacts in noise)

| `kind`           | params                              | implementation                                                 |
|------------------|-------------------------------------|----------------------------------------------------------------|
| `moving_average` | `window` (samples)                  | `pandas.Series.rolling(window, center=True, min_periods=1)`    |
| `lowpass`        | `cutoff` (∈ (0,1) of Nyquist), `order` (default 4) | Butterworth + `scipy.signal.filtfilt` (zero-phase)             |

After smoothing, the noise is renormalised so its empirical std matches the
configured `std` — `std` stays the meaningful knob regardless of `window` /
`cutoff`.

## Adding a new transform

1. Subclass `Transform` in `synthesize.py` and implement `apply(series, rng)`.
2. Register it in the `TRANSFORMS` dict.
3. Reference it from a `syn_cfg_*.yaml` via `{type: <new_name>, ...}`.

## Adding a new domain (e.g. household consumption)

1. Add a top-level block (e.g. `hh_consumption:`) to each `syn_cfg_*.yaml`
   listing its columns and transforms.
2. Extend `SynCfg` in `synthesize.py` with the new field and pass the
   matching file list through `run_synthesis(...)`.
3. Wire the file list in `pipelines.run_synthesize` from the appropriate
   pipeline's outputs.

## Running

Pipeline (final step over all preprocessing outputs):

```bash
python preprocessing/data_setup.py --synthesize
sbatch slurm_scripts/slurm_data_setup.sh --synthesize
```

Standalone, on a single CSV (handy for debugging a config):

```bash
python preprocessing/synthesize.py --domain weather \
    --input data/weather/dwd/preprocessed/2023_merged_04177.csv

python preprocessing/synthesize.py --domain price \
    --input data/e_price/awattar/price_data_2023.csv \
    --only syn_cfg_2_pos
```

`--only` restricts the run to a subset of `active_configs` from the top
config. `--config` overrides the path to the top-level config.

## Output naming

```
<input_stem>_<syn_cfg_name>.csv
```

For example:
```
data/weather/dwd/preprocessed/2023_merged_04177.csv
  → 2023_merged_04177_syn_cfg_1_neg.csv
  → 2023_merged_04177_syn_cfg_1_pos.csv
  → 2023_merged_04177_syn_cfg_2_neg.csv
  → 2023_merged_04177_syn_cfg_2_pos.csv
  → 2023_merged_04177_syn_cfg_3_neg.csv
  → 2023_merged_04177_syn_cfg_3_pos.csv
```

## Noise std targets

The per-column `gaussian_noise.std` values are sized at the *raw* scale of each
input — roughly the measurement uncertainty of the underlying sensor / market
tick — and held constant across all `syn_cfg_*.yaml` presets so that the only
difference between presets is the `constant_shift` offsets:

| column              | std    | rationale                                                |
|---------------------|--------|----------------------------------------------------------|
| `baseprice`         | 0.3    | ct/kWh — tick-size jitter on hourly day-ahead prices     |
| `temp_amb`          | 0.5    | °C — DWD air-temperature sensor accuracy                 |
| `avg_wind_speed`    | 0.3    | m/s — DWD anemometer accuracy                            |
| `direct_sun_shine`  | 50.0   | W/m² — pyranometer direct-component noise floor          |
| `diff_sun_shine`    | 33.0   | W/m² — diffuse-component noise floor                     |
| `hh_consumption_kW` | 0.05   | kW — baseload jitter for a single SFH (typical 0.1-3 kW) |

`sun_shine` is *not* noised directly: `synthesize.py` reconstructs it after the
column-wise transforms so the additive identity stays true in the output CSV
(DWD: `sun_shine = direct_sun_shine + diff_sun_shine`; Zenodo: alias of
`direct_sun_shine`).

## Constant shift ladder

Each `syn_cfg_*_{neg,pos}.yaml` preset applies a `constant_shift` per column on
top of the shared noise budget. Magnitudes follow a weak / medium / strong
ladder (roughly ½× / 1× / 3× the medium tier) and are mirror-symmetric across
`_neg` / `_pos` twins:

| column                       | syn_cfg_1 (weak) | syn_cfg_2 (medium) | syn_cfg_3 (strong) |
|------------------------------|------------------|--------------------|--------------------|
| `baseprice` (ct/kWh)         | ±1.0             | ±2.0               | ±3.0               |
| `temp_amb` (°C)              | ±0.5             | ±1.0               | ±1.5               |
| `avg_wind_speed` (m/s)       | ±0.5             | ±1.0               | ±1.5               |
| `direct_sun_shine` (W/m²)    | ±8.0             | ±17.0              | ±50.0              |
| `diff_sun_shine` (W/m²)      | ±5.0             | ±8.0               | ±17.0              |
| `hh_consumption_kW` (kW)     | ±0.075           | ±0.125             | ±0.25              |

`syn_cfg_0` carries no shifts — it is the noise-only baseline and always uses
the std targets above with `constant_shift` omitted.

## Reproducibility

Each (syn_cfg, file) pair gets a deterministic seed derived from the top
config's `seed`, the index of the active syn_cfg, and the per-domain file
counter. Re-running with the same configs produces byte-identical CSVs.
