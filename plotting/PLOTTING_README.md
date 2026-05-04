# Trajectory Plotting

Plot per-episode trajectory data (states, actions, rewards, energy) from
HDF5 evaluation files produced by the `TrajectoryLoggingCallback`.

## Quick start

```bash
# Plot the latest available ep_metrics (auto-discovers most recent trajectories.hdf5):
python -m plotting

# Plot a specific HDF5 file:
python -m plotting \
    --hdf5 ep_metrics/trajectories/20260310_141900/trajectories.hdf5

# Plot a specific episode:
python -m plotting --episode be574d

# Select by a different metric:
python -m plotting --select-by achieved_reward

# Only produce PNG:
python -m plotting --format png
```

Output is saved to `plotting/out/{episode_id}/` by default.

## CLI reference

```
python -m plotting --help
```

| Flag              | Default                          | Description                                                     |
|-------------------|----------------------------------|-----------------------------------------------------------------|
| `--hdf5`          | latest in `ep_metrics/trajectories/` | Path to `trajectories.hdf5` file. When omitted, auto-discovers the most recent file. |
| `--episode`       | best by `--select-by` metric     | Episode ID (6-char prefix) to plot                              |
| `--select-by`     | `reward_rate`                    | Summary metric for best-episode selection: `reward_rate`, `achieved_reward`, or `cum_E_kWh` |
| `--format`        | `html svg`                       | Output format(s), one or more of: `html`, `png`, `svg`, `pdf`   |
| `--output-dir`    | `plotting/out/<episode_id>/`     | Output directory for generated figures                          |
| `--control-step`  | `300`                            | Control timestep in seconds (5 min default)                     |

Static image formats (`png`, `svg`, `pdf`) require the `kaleido` package:
```bash
pip install kaleido
```

## Episode selection

By default the script scans all episodes in the HDF5 file and selects the
one with the highest `reward_rate` (stored in `summary` attributes).
Use `--select-by` to change the selection metric:

- `reward_rate` (default) -- highest is best
- `achieved_reward` -- highest is best
- `cum_E_kWh` -- lowest is best (minimum energy consumption)

Use `--episode <id>` to bypass auto-selection and plot a specific episode.

## Plot configuration

Domain-specific plotting settings (state groupings, skip keys, action
dimension labels) live in `plotting/plot_config.yaml`. Edit that file
when adding new state sources or infrastructure instead of touching
Python plot modules.

## Figures

The script generates four separate figure groups per episode:

### 1. States (`{episode_id}_states`)

One plot per state variable (or group of related variables) over a 24-hour
day. Scalar state variables (1-D) each get their own figure. Related
variables (e.g. `temp_in_norm` + `desired_temp_in_norm`) are grouped into
a single figure as configured in `plot_config.yaml`. Multi-dimensional
variables with up to 4 columns are plotted with one trace per column;
larger buffers are skipped.

### 2. Actions (`{episode_id}_actions`)

One plot per action dimension. Multi-dimensional actions (e.g. `HP_action`
with `[energy, mode]`) are expanded so each dimension gets its own figure
with labelled names from `plot_config.yaml`. Scalar actions
(`battery_action`, `solar_action`) are single traces.

### 3. Rewards (`{episode_id}_rewards`)

Stacked area chart of per-component reward breakdown (e.g. `temp_reward`,
`economic_reward`) with a bold black line for the total reward overlaid
on top.

### 4. Energy (`{episode_id}_energy`)

Dual y-axis figure with:
- **Bar chart** (left y-axis): instantaneous power draw per step (kW)
- **Line plot** (right y-axis): cumulative energy consumption (kWh)

## Input data

The script reads `trajectories.hdf5` files written by the
`TrajectoryLoggingCallback` during evaluation. The HDF5 schema is documented
in `docs/about_traj_hdf5_export.md`. Each episode group contains:

```
<episode_id>/
├── summary/       (attrs: achieved_reward, reward_rate, cum_E_kWh, ...)
└── trajectory/
    ├── step, reward, cum_E_kWh, step_power_kW   (1-D arrays)
    ├── state/          (one dataset per state variable)
    ├── action/         (one dataset per action variable)
    └── reward_breakdown/  (one dataset per reward component)
```

## Python API

The plotting functions can also be called directly:

```python
from plotting import load_episode, plot_states, generate_all_plots

# Load and inspect data (selects best reward_rate by default)
episode = load_episode("trajectories.hdf5")
print(episode.episode_id)         # 'be574d'
print(episode.seed)               # 42
print(list(episode.states))       # ['temp_in_norm', 'E_price', ...]

# Load by a specific metric
episode = load_episode("trajectories.hdf5", select_by="achieved_reward")

# Generate a single figure
figs = plot_states(episode)
figs[0].show()

# Generate all figures and save (default: html + svg)
paths = generate_all_plots("trajectories.hdf5")

# Generate only png
paths = generate_all_plots("trajectories.hdf5", formats=["png"])
```

## Output structure

```
plotting/out/
└── {episode_id}/
    ├── {episode_id}_states.html      # all state plots in one page
    ├── {episode_id}_actions.html     # all action plots in one page
    ├── {episode_id}_rewards.html
    ├── {episode_id}_energy.html
    └── svgs/                         # static images in format subdirectory
        ├── {episode_id}_states_0.svg
        ├── {episode_id}_states_1.svg
        ├── ...
        ├── {episode_id}_actions_0.svg
        ├── ...
        ├── {episode_id}_rewards.svg
        └── {episode_id}_energy.svg
```

HTML files are placed directly under the episode directory. Static image
formats (`svg`, `png`, `pdf`) go into a subdirectory named after the format
(e.g. `svgs/`, `pngs/`).

TODO noprio VP: seaborn plots needed as well

## Dependencies

- `plotly` (interactive HTML plots)
- `h5py` (HDF5 reading)
- `numpy`
- `kaleido` (only for static image export: png/svg/pdf)
