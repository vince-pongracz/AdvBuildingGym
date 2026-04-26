# Synthetic Dataset Generator

Generates noised / shifted copies of preprocessed price and weather CSVs to
expand the scenario pool used by `DataCombinator`.

## Layout

```
preprocessing/
├── synthesize.py             # transform engine + CLI + pipeline entry point
├── synthesize_config.yaml    # top-level: which syn_cfgs to apply
└── syn_cfgs/
    ├── syn_cfg_1.yaml        # weak  (small negative shifts)
    ├── syn_cfg_2.yaml        # medium (moderate positive shifts)
    ├── syn_cfg_3.yaml        # strong (large positive shifts)
    └── syn_cfg_*.yaml        # ...
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
active_configs: [syn_cfg_1, syn_cfg_2, syn_cfg_3]
```

## Per-level config (`syn_cfg_*.yaml`)

Each level defines, under `price` and `weather`, a per-column transform
pipeline + optional clip bounds:

```yaml
name: syn_cfg_2

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

The three preset configs intentionally share noise levels and only differ in
their `constant_shift` values, so the synthesised datasets cover three
"climate / market" offsets at the same noise budget.

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
    --only syn_cfg_2
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
  → 2023_merged_04177_syn_cfg_1.csv
  → 2023_merged_04177_syn_cfg_2.csv
  → 2023_merged_04177_syn_cfg_3.csv
```

## Open TODOs

- **Tune weather noise std (TODO VP 2026-03-10)** — the per-column `gaussian_noise.std`
  values in the three syn_cfgs are rough initial guesses scaled to each
  variable's typical magnitude. They should be tuned against DWD sensor
  measurement uncertainty and the desired augmentation strength
  (`temp_amb` 0.5 °C, `avg_wind_speed` 0.3 m/s, `sun_shine` 5.0 J/cm²,
  `direct_sun_shine` 3.0 J/cm², `diff_sun_shine` 2.0 J/cm²).

## Reproducibility

Each (syn_cfg, file) pair gets a deterministic seed derived from the top
config's `seed`, the index of the active syn_cfg, and the per-domain file
counter. Re-running with the same configs produces byte-identical CSVs.
