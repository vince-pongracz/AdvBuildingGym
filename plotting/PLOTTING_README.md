# Plotting

Two independent sub-packages live under `plotting/`:

- [`plotting.traj_plotting`](traj_plotting/) — per-episode trajectory plots
  from `trajectories.hdf5` files produced by the `TrajectoryLoggingCallback`
  during evaluation / training.
- [`plotting.data_plotting`](data_plotting/) — dataset-level visualisations
  (weather, energy prices, household consumption, EV profiles, synthesised
  variants) sourced directly from CSV files in `data/`.

Configs:

- [plotting/config/traj_plot_config.yaml](config/traj_plot_config.yaml) — domain
  config for trajectory plots (state groupings, skip keys, figure size).
- [plotting/config/data_plot_config.yaml](config/data_plot_config.yaml) —
  data-source definitions and figure sizing for data plots.

Default output roots (overridable in the YAMLs):

- Trajectory plots → `plotting/out/traj_plots/<episode_label>/`
- Data plots → `plotting/out/data_plots/`

Static formats (`png`, `svg`, `pdf`) require `kaleido` (auto-fetched
Chrome at first use for the trajectory pipeline). HTML output has no
extra dependency.

---

## 1. Trajectory plotting (`plotting.traj_plotting`)

Entry point:

```bash
python -m plotting.traj_plotting                    # short form, equivalent to:
python -m plotting.traj_plotting.trajectory_plot
```

Per episode, six figure groups are emitted (each a list of plotly figures):

| Group                | Source                              | What it shows |
|----------------------|-------------------------------------|----------------|
| `states`             | `s_*` / `ctxt_*` obs keys           | Normalised observations; groupings from `traj_plot_config.yaml` |
| `actions`            | Dict action entries (`a_*`)         | One figure per action dimension |
| `raw_policy_actions` | `raw_policy_*` (pre-rescale tanh)   | Raw policy outputs in `[-1, 1]` |
| `rewards`            | per-component `reward_breakdown`    | Stacked area + total reward line |
| `energy`             | `net_power_kW` / `cum_E_kWh`        | Bar (power) + line (cumulative kWh) dual y-axis |
| `raw`                | `raw_*` physical values             | Unnormalised temperatures, prices, kW, etc. |

### CLI options

| Flag             | Default                          | Description |
|------------------|----------------------------------|-------------|
| `--hdf5`         | latest in `ep_metrics/trajectories/` | Path to `trajectories.hdf5`. |
| `--episode`      | best by `--select-by`            | Plot a specific episode id (HDF5 group name). |
| `--all-episodes` | off                              | Plot every episode in the file (one subdir per episode). Mutex with `--episode`. |
| `--select-by`    | `achieved_reward`                | Auto-pick metric. One of `achieved_reward`, `cum_E_kWh` (lowest is best for this one). |
| `--output-dir`   | `plotting/out/traj_plots/ep_<id>/` | Output root. With `--all-episodes` this is the parent dir. |
| `--format`       | `html svg`                       | One or more of `html`, `png`, `svg`, `pdf`. |
| `--control-step` | `300`                            | Control step duration in seconds (used for time-axis). |

### Examples (one per mode / option)

```bash
# Latest HDF5, best episode (by achieved_reward), html + svg
python -m plotting.traj_plotting

# Pick a specific HDF5 file
python -m plotting.traj_plotting --hdf5 ep_metrics/trajectories/20260310_141900/trajectories.hdf5

# Plot a specific episode by its HDF5 group id
python -m plotting.traj_plotting --episode be574d12

# Plot every episode in the file (one subdir per episode)
python -m plotting.traj_plotting --all-episodes

# Auto-select episode by cum_E_kWh instead
python -m plotting.traj_plotting --select-by cum_E_kWh

# Auto-select episode by lowest cumulative energy
python -m plotting.traj_plotting --select-by cum_E_kWh

# Custom output directory
python -m plotting.traj_plotting --output-dir /tmp/my_traj_plots

# Only HTML (skip kaleido / Chrome download)
python -m plotting.traj_plotting --format html

# Only static PNG
python -m plotting.traj_plotting --format png

# Multiple formats at once
python -m plotting.traj_plotting --format html png svg pdf

# Non-default control step (e.g. 60 s)
python -m plotting.traj_plotting --control-step 60

# SLURM wrapper — forwards all flags
sbatch slurm_scripts/slurm_plot_trajectory.sh --hdf5 path/to/trajectories.hdf5 --all-episodes --format html png
```

### Python API

```python
from plotting.utils import load_episode
from plotting.traj_plotting.trajectory_plot import generate_all_plots
from plotting.traj_plotting.plot_states import plot_states

episode = load_episode("trajectories.hdf5", select_by="achieved_reward")
figs = plot_states(episode)
figs[0].show()

paths = generate_all_plots(
    hdf5_path="trajectories.hdf5",
    episode_id=None,             # None = best by select_by
    formats=["html", "png"],
    select_by="achieved_reward",
)
```

### Output layout

```
plotting/out/traj_plots/
└── ep_<episode_id>/
    ├── ep_<id>_states.html          # all state figures in one page
    ├── ep_<id>_actions.html
    ├── ep_<id>_raw_policy_actions.html
    ├── ep_<id>_rewards.html
    ├── ep_<id>_energy.html
    ├── ep_<id>_raw.html
    └── svgs/                        # static formats nest into a <fmt>s/ subdir
        ├── ep_<id>_states_0.svg
        ├── ep_<id>_states_1.svg
        └── ...
```

`generate_all_plots` is also invoked by `run_eval_ray.py --plot` so the
same figure set lands inside the timestamped `eval_results/.../` dir
without running this CLI manually.

TODO noprio VP: seaborn plots needed

---

## 2. Data plotting (`plotting.data_plotting`)

Three scripts, each a separate module, all sharing
[data_plot_config.yaml](config/data_plot_config.yaml) and a common CLI
preamble (`--config`, `--format`, `--warn-future-data`).

The shared options (apply to **all three** scripts):

| Flag                  | Default                                   | Description |
|-----------------------|-------------------------------------------|-------------|
| `--config`            | `plotting/config/data_plot_config.yaml`   | Path to the data plot config. |
| `--format`            | `html`                                    | One or more of `html`, `png`, `svg`, `pdf`. |
| `--warn-future-data`  | off                                       | Emit `No data for <date>` warnings even for future dates. |

Each weather/price section in the YAML carries multiple datasets (e.g.
weather: `dwd`, `zenodo`; price: `awattar`, `e_charts`, `test1`) and a
`use:` selector for the single-day script. Any sibling `*_syn_cfg_*.csv`
synthesised file produced by `preprocessing/synthesize.py` is picked up
automatically and overlaid with a distinct colour.

### 2a. `plot_day_data` — one-or-more concrete dates

Plots weather / price / desired-temp / household-consumption / EV-schedule
for explicit calendar dates. Uses the **single dataset** selected by
`weather.use` / `price.use` in the config.

Positional arg: one or more dates as `YYYY-MM-DD`.

Modes / extra flags:

| Flag        | Default | Description |
|-------------|---------|-------------|
| `--days N`  | `1`     | If a single date is given, expand into N consecutive days. Ignored for multi-date input. |
| `--stat`    | off     | Multi-day mode only: show mean + ±1 std band, hide per-day traces. |

Examples (covering every option / mode):

```bash
# Single day
python -m plotting.data_plotting.plot_day_data 2020-07-15

# Multiple explicit dates (overlaid + mean + std band)
python -m plotting.data_plotting.plot_day_data 2020-07-15 2020-08-01 2021-01-10

# Start date + N consecutive days
python -m plotting.data_plotting.plot_day_data 2020-07-15 --days 7

# Same but only stat summary (no per-day spaghetti)
python -m plotting.data_plotting.plot_day_data 2020-07-15 --days 30 --stat

# Multiple output formats
python -m plotting.data_plotting.plot_day_data 2020-07-15 --days 3 --format html png

# Static-only export
python -m plotting.data_plotting.plot_day_data 2020-07-15 --days 7 --format svg

# Use an alternative config
python -m plotting.data_plotting.plot_day_data 2020-07-15 --config plotting/config/data_plot_config.yaml

# Enable future-date warnings
python -m plotting.data_plotting.plot_day_data 2027-01-01 --warn-future-data
```

Output: `plotting/out/data_plots/day_YYYYMMDD.<fmt>` or
`days_YYYYMMDD_n<N>[_stat].<fmt>`.

### 2b. `plot_monthly_overview` — all months × all years × all dataset combos

For every (weather × price) dataset combo found in the YAML, generates:

1. **Per year-month**: one figure per `(year, month)` with all days in
   that month overlaid (mean + std band; `stat_only=True` by default).
2. **Cross-year per calendar month**: one figure per calendar month
   collecting *every* occurrence across `--start-year`..`--end-year`.

Y-axes are shared across all months/years per column, so figures are
visually comparable.

Extra flags:

| Flag           | Default | Description |
|----------------|---------|-------------|
| `--start-year` | `2016`  | First year (inclusive). |
| `--end-year`   | `2026`  | Last year (inclusive). |

Examples:

```bash
# Full sweep (2016-2026, every combo, html only)
python -m plotting.data_plotting.plot_monthly_overview

# Restrict year range
python -m plotting.data_plotting.plot_monthly_overview --start-year 2020 --end-year 2024

# Multiple formats
python -m plotting.data_plotting.plot_monthly_overview --format html png

# Alternative config / future-date warnings
python -m plotting.data_plotting.plot_monthly_overview --config plotting/config/data_plot_config.yaml --warn-future-data
```

Output: `plotting/out/data_plots/monthly_overview/<weather>_<price>/per_month/<YYYY-MM>.<fmt>`
and `.../cross_year/all_years_<MM>_<Mon>.<fmt>`.

### 2c. `plot_cross_year_combined` — all datasets overlaid per month

For each of the 12 calendar months, builds figures where **every**
weather dataset (and every synthesised variant) gets its own colour-coded
stat band on a shared axis; same for price. Profile data is overlaid
once per month (date-independent).

Same year-range options as `plot_monthly_overview`:

| Flag           | Default | Description |
|----------------|---------|-------------|
| `--start-year` | `2016`  | First year (inclusive). |
| `--end-year`   | `2026`  | Last year (inclusive). |

Examples:

```bash
# Default sweep
python -m plotting.data_plotting.plot_cross_year_combined

# Restrict year range
python -m plotting.data_plotting.plot_cross_year_combined --start-year 2020 --end-year 2024

# Multiple formats
python -m plotting.data_plotting.plot_cross_year_combined --format html png

# Static svg + future-date warnings + custom config
python -m plotting.data_plotting.plot_cross_year_combined --format svg --warn-future-data --config plotting/config/data_plot_config.yaml
```

Output: `plotting/out/data_plots/cross_year_combined/all_years_<MM>_<Mon>_combined.<fmt>`.

### SLURM wrapper

```bash
# Run both data-plotting scripts in parallel
sbatch slurm_scripts/slurm_data_vis.sh

# Run a single target (cross_year | monthly)
sbatch slurm_scripts/slurm_data_vis.sh cross_year
sbatch slurm_scripts/slurm_data_vis.sh monthly

# Forward extra args to all scripts (after the target, or directly if no target)
sbatch slurm_scripts/slurm_data_vis.sh monthly --start-year 2022 --end-year 2024 --format html png
sbatch slurm_scripts/slurm_data_vis.sh --format html png
```

Logs land in `slurm_logs/data_vis/`.

---

## Dependencies

- `plotly` — interactive figures (HTML output)
- `h5py` + `numpy` — trajectory HDF5 reading
- `pandas` — data-plotting CSV ingestion
- `pyyaml` — config loading
- `kaleido` — static image export (`png` / `svg` / `pdf`); the trajectory
  pipeline auto-downloads Chrome to `~/.cache` on first static export.
