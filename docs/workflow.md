# Workflow: Data Setup, Training, Evaluation, Plotting

End-to-end pipeline for running RL experiments on AdvBuildingGym.

```
data setup  ──>  train  ──>  evaluate  ──>  plot
```

---

## 1. Data Setup

Fetch and preprocess electricity prices and weather data into 5-minute resolution CSVs
(raw physical units). Normalisation to agent-friendly ranges happens at runtime in the
statesources, not here -- statesources expose the normalisation constants as context variables, so the policy can see this and make decisions on this.

**Script:** `preprocessing/data_setup.py`
**SLURM:** `sbatch slurm_scripts/slurm_data_setup.sh [OPTIONS]`

```bash
# Full pipeline (prices + weather)
python preprocessing/data_setup.py

# Prices only, specific years
python preprocessing/data_setup.py --skip-weather --years 2023 2024 2025

# Reuse existing raw files, then synthesise dataset variants
python preprocessing/data_setup.py --skip-weather --skip-price-fetch \
    --years 2023 --raw-price-files data/e_price/awattar/2023_prices.csv --synthesize
```

**Key arguments:**

| Flag | Purpose |
|------|---------|
| `--years YEAR [YEAR ...]` | Target years (default: 2017-2026) |
| `--price-source {awattar,energy-charts}` | Price API source |
| `--skip-weather` | Skip weather (WPuQ and DWD) pipeline |
| `--skip-price-fetch` | Skip API calls, use local CSVs |
| `--synthesize` | Run synthetic dataset generation as the final pipeline step |

**Output:** raw 5-minute CSVs in `data/`:
- `data/e_price/awattar/price_data_<YEAR>.csv` (baseprice in ct/kWh)
- `data/e_price/e_charts/price_data_<YEAR>.csv` (baseprice in ct/kWh)
- `data/weather/dwd/preprocessed/<YEAR>_merged_04177.csv` (temp in °C, wind in m/s, etc.)
- `data/ev_usage_profiles/ev_*.csv` (pre-existing)

### 1.1 Synthetic Dataset Generation

Synthesis produces noised / shifted copies of the preprocessed price and weather CSVs,
expanding the scenario pool used by `DataCombinator`. Runs as the final pipeline step
when `--synthesize` is passed, or standalone via `preprocessing/synthesize.py`. See
[preprocessing/SYNTHESIZE_README.md](../preprocessing/SYNTHESIZE_README.md) for the
full reference.

**Configuration:**
- Top level: `preprocessing/synthesize_config.yaml` — base seed and the list of
  active per-level configs.
- Per level: `preprocessing/syn_cfgs/syn_cfg_*.yaml` — for each domain (`price`,
  `weather`, `hh_consumption`) and each column, a transform pipeline (Gaussian noise
  with optional smoothing, constant shifts, linear scaling) plus optional clip
  bounds. The shipped presets share noise levels and only differ in their
  `constant_shift` values, covering symmetric `_pos` / `_neg` pairs at three
  intensities plus a noise-only baseline (`syn_cfg_0`).

```yaml
# preprocessing/synthesize_config.yaml
seed: 42
syn_cfg_dir: preprocessing/syn_cfgs
active_configs: [syn_cfg_1_neg, syn_cfg_1_pos, syn_cfg_2_neg, syn_cfg_2_pos, syn_cfg_3_neg, syn_cfg_3_pos]
```

```yaml
# preprocessing/syn_cfgs/syn_cfg_2_pos.yaml (excerpt)
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

**Integrated usage** (via `data_setup.py`):

```bash
# Full pipeline with synthesis as final step
python preprocessing/data_setup.py --synthesize

# Synthesise from existing preprocessed files only
python preprocessing/data_setup.py --skip-weather --skip-price-fetch \
    --raw-price-files data/e_price/awattar/2023_prices.csv --synthesize
```

**Standalone usage** (via `preprocessing/synthesize.py`, handy for debugging a config):

```bash
# Run all active configs on a single weather CSV
python preprocessing/synthesize.py --domain weather \
    --input data/weather/dwd/preprocessed/2023_merged_04177.csv

# Restrict to a subset of active_configs
python preprocessing/synthesize.py --domain price \
    --input data/e_price/awattar/price_data_2023.csv --only syn_cfg_2
```

| Flag | Purpose |
|------|---------|
| `--domain {price,weather}` | Which domain block to use from each `syn_cfg_*.yaml` |
| `--input PATH` | Source CSV (one of the preprocessed files) |
| `--only CFG [CFG ...]` | Restrict to a subset of `active_configs` |
| `--config PATH` | Override the top-level synthesise config |

**Output naming:** each input `<stem>.csv` produces one
`<stem>_<syn_cfg_name>.csv` per active config, written next to the source file.
Each (syn_cfg, file) pair gets a deterministic seed derived from the base seed,
so re-running with the same configs is reproducible.

**Training integration:** set `include_synthesized: true` in
`configs/train_data_combinator_config.yaml` so `discover_synthetic_scenarios`
auto-discovers `*_syn_cfg_*.csv` files and adds them to the scenario pool
(see section 2.3).

---

## 2. Training

### 2.1 Environment config — `configs/env/env_test1_{small,mid,large}.yaml`

The env config defines the **environment topology**: which infrastructures and
statesources are instantiated and their parameters, plus episode timing. It is the
single entry point passed to both training and evaluation via `--load-config`.

The file at `configs/env/<name>.yaml` is a thin **wrapper** that references three
sibling YAMLs — concerns are split so an infra topology, statesource bundle, or
timing constants can be reused independently:

```yaml
# configs/env/env_test1_small.yaml  (the wrapper)
env_config_name: env_test1_small               # used as the checkpoint dir name
infras: configs/infras/test1_small.yaml        # controllable devices
statesources: configs/statesources/default.yaml # observation providers
env_meta: configs/env_meta/default.yaml        # EPISODE_LENGTH, control_step
```

| Referenced file | Top-level keys | What it controls |
|---|---|---|
| `configs/env_meta/*.yaml` | `EPISODE_LENGTH`, `control_step` | Steps per episode (`288` = 24 h) and seconds per step (`300`) |
| `configs/infras/*.yaml` | `infras: [...]` | Controllable devices: HP, BatteryTremblay/Linear, LinearEVCharger, SolarPanel, WindTurbine, HouseholdEnergyConsumers — each entry is `{class, name, ...params}` |
| `configs/statesources/*.yaml` | `statesources: [...]` | Observation providers: WeatherDataSource, EnergyPriceDataSource, InsideTemperature, DesiredUserEnergyNeed, BuildingHeatLoss (carries the 1R1C envelope params `K`, `mC`), EVState, OperatorEnergyControl |

**What is NOT in the env config:**
- **Rewards** — composed separately via `configs/reward_cfg/reward_schedule_*.yaml`
  (passed with `--reward-schedule`); see section 2.5. There is no `rewards:` key
  in the env wrapper.
- **Data scenarios** — driven by `configs/data_scheduler/*.yaml` (`--data-config`).
- **Building envelope (`K`, `mC`)** — these are parameters of the `BuildingHeatLoss`
  statesource, not a top-level `building_props` block. Edit them in the
  statesources YAML.

`EnvConfigManager.load(wrapper_path)` resolves the three referenced paths
relative to the project root (so they can be repo-relative like
`configs/infras/...`), parses each, and returns a populated `EnvConfig` whose
`create_infras()` / `create_statesources()` factories produce fresh per-worker
instances. Round-trip is symmetric: `EnvConfigManager.save(...)` writes all four
files back out.

Load with `--load-config configs/env/env_test1_{small,mid,large}.yaml` on either
`run_train_ray.py` or `run_eval_ray.py`. Use the same wrapper for eval as for
training so the topology matches the checkpoint.

### 2.2 Training hyperparameters — `configs/training_param_config.yaml`

Algorithm-agnostic and algorithm-specific RL hyperparameters. Always loaded
automatically by `run_train_ray.py`.

```yaml
common:
  learning_rate: 3.0e-4
  seed: 42
  episode_lookback_horizon_steps: 120   # 10-hour temporal window; auto-raised to max(|hst.offsets|) if smaller
  max_episodes_to_run: 7000

ppo:
  episodes_per_iteration: 25            # episodes collected per policy update
  minibatch_size: 64
  num_epochs: 20

sac:
  replay_batch_size: 256                # transitions per gradient step
  episodes_to_keep_in_replay_buffer: 100
```

CLI `--seed` overrides `common.seed`. Other values are edited in the YAML directly.

### 2.3 Data combinator — `configs/train_data_combinator_config.yaml`

Controls **scenario scheduling** during training: which weather/price/EV-profile
combinations the agent sees, and how often they rotate.

```yaml
seed: 42
shuffle: true
swap_every_n_episodes: 2        # rotate data variant every 2 episodes
mode: cycle                     # round-robin through variants
day: random                     # random day within each scenario

years: [2018, 2019, 2020, 2021, 2022, 2023, 2024]
include_synthesized: true

scenario_sources:               # Cartesian product across years
  - weather: data/weather/dwd/preprocessed/{year}_merged_04177.csv
    E_price: data/e_price/awattar/price_data_{year}.csv
  - weather: data/weather/dwd/preprocessed/{year}_merged_04177.csv
    E_price: data/e_price/e_charts/price_data_{year}.csv

variable:                       # independent Cartesian factor
  ev_schedule:
    - data/ev_usage_profiles/ev_0.csv
    - data/ev_usage_profiles/ev_1.csv
    # ... (6 profiles total)
```

Total variant pool = (2 sources x 7 years) x 6 EV profiles = 84 combinations.

### 2.4 Running training

**Script:** `run_train_ray.py`
**SLURM:** `sbatch slurm_scripts/slurm_train_ray.sh [OPTIONS]`

SLURM resources: 4 CPUs, 1 GPU, 10 min (increase `--time` for real runs).

```bash
# Default: PPO, 7000 episodes (from config), seed 42, optimise reward_rate
sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo

# PPO with fewer episodes (CLI override)
sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --episodes 3500

# SAC with custom seed and more episodes
sbatch slurm_scripts/slurm_train_ray.sh --algorithm sac --seed 18 --episodes 5000

# Load a custom environment config
sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --load-config configs/env/env_test1_{s/m/l}.yaml --episodes 3500

# Enable trajectory logging during training eval
sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --episodes 3500 --log-trajectories
```

**Key arguments:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--algorithm {ppo,sac}` | `ppo` | RL algorithm |
| `--episodes N` | from config (`7000`) | Total training episodes (primary stop criterion) |
| `--seed N` | `42` (from YAML) | Random seed |
| `--metric {reward_rate,achieved_reward,episode_return_mean}` | `reward_rate` | Optimisation metric |
| `--checkpoint-frequency-episodes N` | `20` | Save checkpoint every N episodes |
| `--load-config PATH` | — (**required**) | Environment YAML to load; `env_config_name` inside sets the checkpoint dir name |
| `--data-config PATH` | `configs/train_data_combinator_config.yaml` | Data combinator YAML |
| `--log-trajectories` | off | Save per-step trajectory JSON during eval |

**What the script does:**

1. Loads `training_param_config.yaml` + environment config + data combinator config
2. Auto-detects SLURM resources (`SLURM_CPUS_PER_TASK`, `CUDA_VISIBLE_DEVICES`)
3. Initialises Ray with detected resources
4. Builds RLlib algorithm config (PPO or SAC) via `select_model()` + `common_model_setup()`
5. Runs Ray Tune with checkpointing and metric-based best-model tracking
6. Logs best trial results

**Output:**

| Artifact | Path |
|----------|------|
| Ray trial directory | `models/<config>/ray/<algo>/<run_name>/` |
| Best checkpoint | `models/<config>/ray/<algo>/checkpoints_<run>/best_model_ep*.ckpt` |
| Parameter space | `<trial_dir>/param_space.json` |
| TensorBoard logs | `<trial_dir>/` (use `tensorboard --logdir`) |
| SLURM logs | `slurm_logs/train/slurm-train-ray-<JOBID>.{out,err}` |

---

## 3. Evaluation

### 3.1 Data combinator — `configs/eval_data_combinator_config.yaml`

Same structure as the training data combinator (section 2.3), with key differences
for **deterministic evaluation**:

| Setting | Training | Evaluation |
|---------|----------|------------|
| `shuffle` | `true` | `false` |
| `swap_every_n_episodes` | `2` | `1` (new variant each episode) |
| `mode` | `cycle` | `cycle` (deterministic round-robin) |
| `day` | `random` | `each` (sequential days) |
| `years` | 2018-2024 | 2025-2026 (held-out) |

### 3.2 Running evaluation

**Script:** `run_eval_ray.py`
**SLURM:** `sbatch slurm_scripts/slurm_eval_ray.sh [OPTIONS]`

SLURM resources: 2 CPUs, no GPU (CPU inference is sufficient).

```bash
# Evaluate best PPO checkpoint (auto-discovered)
sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --episodes 10

# Evaluate specific checkpoint
sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --checkpoint models/test1/ray/ppo/best_model_ep100

# Evaluate and generate plots in one step
sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --episodes 10 --plot

# Use eval-specific data combinator for held-out years
sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --episodes 20 --data-config
```

**Key arguments:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--algorithm {ppo,sac}` | `ppo` | Algorithm (determines checkpoint search path) |
| `--checkpoint PATH` | auto-detect best | Explicit checkpoint path |
| `--episodes N` | `10` | Number of evaluation episodes |
| `--seed N` | `42` | Random seed |
| `--output-dir PATH` | `eval_results` | Results directory |
| `--load-config PATH` | — | Environment YAML |
| `--data-config [PATH]` | — | Eval data combinator (bare flag uses `configs/eval_data_combinator_config.yaml`) |
| `--data-day SPEC` | — | Override day: `random`, `each`, or `2022-07-15` |
| `--plot` | off | Plot best episode trajectory after eval |
| `--plot-all` | off | Plot all episodes |
| `--log-trajectories / --no-log-trajectories` | on | Save per-step trajectory data |

**What the script does:**

1. Loads environment config and (optionally) eval data combinator
2. Resolves checkpoint path (auto-discovers best if not provided)
3. Initialises Ray (CPU-only), loads RLModule from checkpoint
4. Runs episodes: reset → infer → step → collect trajectories
5. Aggregates metrics: `achieved_reward`, `reward_rate`, `cum_E_kWh`
6. Saves results JSON + trajectory HDF5
7. Optionally generates plots

**Output:**

| Artifact | Path |
|----------|------|
| Results summary | `eval_results/<TIMESTAMP>_eval/eval_results.json` |
| Trajectory archive | `eval_results/<TIMESTAMP>_eval/trajectories.hdf5` |
| Per-episode JSON | `eval_results/<TIMESTAMP>_eval/trajectories/<ep_id>_trajectory.json` |
| Plots (if `--plot`) | `eval_results/<TIMESTAMP>_eval/plots/` |
| SLURM logs | `slurm_logs/eval/slurm-eval-ray-<JOBID>.{out,err}` |

---

## 4. Plotting

Three plotting workflows are available: **trajectory plotting** (standalone, from HDF5),
**eval plotting** (integrated into the evaluation script), and **data plotting** (raw
weather/price data inspection).

### 4.1 Trajectory Plotting (standalone)

**Script:** `python -m plotting.traj_plotting`
**SLURM:** `sbatch slurm_scripts/slurm_plot_trajectory.sh [OPTIONS]`

SLURM resources: 1 CPU, no GPU.

```bash
# Auto-discover latest HDF5, plot best episode
python -m plotting.traj_plotting

# Specific HDF5 file
python -m plotting.traj_plotting --hdf5 eval_results/20260324_eval/trajectories.hdf5

# Select best episode by a specific metric
python -m plotting.traj_plotting --select-by achieved_reward

# Specific episode, multiple output formats
python -m plotting.traj_plotting --episode be574d --format html svg png

# Custom output directory
python -m plotting.traj_plotting --output-dir my_plots/
```

**Key arguments:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--hdf5 PATH` | auto-discover latest | Path to `trajectories.hdf5` |
| `--episode ID` | best by `--select-by` | Episode ID to plot |
| `--select-by METRIC` | `reward_rate` | Metric for best-episode selection |
| `--format FMT [...]` | `html svg` | Output formats: `html`, `png`, `svg`, `pdf` |
| `--output-dir PATH` | `plotting/out/<episode_id>/` | Output directory |
| `--control-step N` | `300` | Timestep in seconds (for x-axis) |

**Configuration:** `plotting/config/traj_plot_config.yaml` — domain-specific rendering
settings. Edit this (not Python code) when adding new statesources or infrastructure.

```yaml
states:
  skip_keys: [E_price_max, sim_hour, _temp_abs_max]   # omit from plots
  grouped_keys:                                         # share a subplot
    - [battery_pct, ]
    - [temp_in_norm, desired_temp_in_norm, temp_out_norm]
    - [ev_soc, ev_target_soc]
  mask_when_disconnected:
    condition_key: ev_connected
    keys: [ev_soc, ev_target_soc, ev_soc_hist, ev_charge_to_target_hrs_norm]

actions:
  dim_labels:
    HP_action: [energy, mode]                           # human-readable names

raw:
  grouped_keys:
    - [temp_out_raw, temp_in_raw, desired_temp_in_raw]
  skip_keys: [E_price_max_raw]

output:
  dir: plotting/out/traj_plots
```

**Generated figures per episode:**

| Figure | Content |
|--------|---------|
| `<ep>_states` | Normalised observation traces (temperature, SoC, price, ...) |
| `<ep>_actions` | Per-action-dimension traces (HP energy/mode, battery, EV) |
| `<ep>_rewards` | Stacked area of per-component rewards + total overlay |
| `<ep>_energy` | Bar chart (instantaneous kW) + line (cumulative kWh) |
| `<ep>_raw` | Denormalised physical values (temperature in C, price in EUR) |

**Output:** `plotting/out/traj_plots/` (or `--output-dir`).

### 4.2 Eval Plotting (integrated)

The evaluation script `run_eval_ray.py` can generate trajectory plots directly after
evaluation completes, without a separate plotting step.

```bash
# Plot best episode after evaluation
sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --episodes 10 --plot

# Plot all episodes after evaluation
sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --episodes 10 --plot-all
```

| Flag | Purpose |
|------|---------|
| `--plot` | Plot the best episode's trajectory (implies `--log-trajectories`) |
| `--plot-all` | Plot all episodes' trajectories (implies `--log-trajectories`) |

Both flags automatically enable trajectory logging. Plots are saved to
`eval_results/<TIMESTAMP>_eval/plots/` — one subdirectory per episode when using
`--plot-all`. The same figure set (states, actions, rewards, energy, raw) is generated
as in standalone trajectory plotting.

### 4.3 Data Plotting (raw input data)

Visualise raw weather and energy-price CSV data for one or more calendar days.
Useful for inspecting data quality, comparing days, and verifying preprocessing output.
Each day is overlaid as a separate trace; multi-day plots add a mean curve with
+/- 1 std-dev band.

**Script:** `python -m plotting.traj_plotting.data_plotting.plot_day_data`

```bash
# Single day
python -m plotting.traj_plotting.data_plotting.plot_day_data 2020-07-15

# Multiple explicit dates (overlaid)
python -m plotting.traj_plotting.data_plotting.plot_day_data 2020-07-15 2020-08-01 2021-01-10

# Start date + N consecutive days
python -m plotting.traj_plotting.data_plotting.plot_day_data 2020-07-15 --days 7

# Statistical summary only (mean + std band, hide individual traces)
python -m plotting.traj_plotting.data_plotting.plot_day_data 2020-07-15 --days 7 --stat

# Custom config and output formats
python -m plotting.traj_plotting.data_plotting.plot_day_data 2020-07-15 --config plotting/config/data_plot_config.yaml --format html png
```

**Key arguments:**

| Flag | Default | Purpose |
|------|---------|---------|
| `dates` (positional) | — | One or more dates (YYYY-MM-DD) |
| `--days N` | `1` | Plot N consecutive days from the first date (ignored when multiple dates given) |
| `--stat` | off | Show only mean + std band, hide individual day traces |
| `--config PATH` | `plotting/config/data_plot_config.yaml` | Data plot config |
| `--format FMT [...]` | `html` | Output formats: `html`, `png`, `svg`, `pdf` |

**Configuration:** `plotting/config/data_plot_config.yaml` — selects weather source
(DWD or Zenodo) and price source (aWATTar, Energy-Charts, or test1) via the `use` key
in each section.

**Generated figures:**

| Figure | Content |
|--------|---------|
| Weather panels | One figure per variable: temperature (°C), humidity (%), wind speed (m/s), irradiance (W/m²) |
| Energy price | Price traces (ct/kWh) with area fill (single day) or overlay (multi-day) |

**Output:** `plotting/out/data_plots/`.

---

## Quick Reference: Full Pipeline

```bash
# 1. Data setup (one-time, or when adding new years)
sbatch slurm_scripts/slurm_data_setup.sh --years 2023 2024 2025

# 2. Train
sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --episodes 3500 --seed 42

# 3. Evaluate on held-out data
sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --episodes 20 --data-config

# 4. Plot
sbatch slurm_scripts/slurm_plot_trajectory.sh --select-by reward_rate --format html svg
```

**Local (no SLURM):** replace `sbatch slurm_scripts/slurm_*.sh` with the Python
command directly (activate the venv first: `source ../adv_env/bin/activate`).

---

## Log Files

All SLURM logs are in `slurm_logs/`. The `.err` file is the primary source for
troubleshooting (Python logging goes to stderr). Higher job ID = more recent run.

| Stage | Log path |
|-------|----------|
| Data setup | `slurm_logs/data_setup/slurm-data-setup-<JOBID>.{out,err}` |
| Training | `slurm_logs/train/slurm-train-ray-<JOBID>.{out,err}` |
| Evaluation | `slurm_logs/eval/slurm-eval-ray-<JOBID>.{out,err}` |
| Plotting | `slurm_logs/plotting/slurm-plot-trajectory-<JOBID>.{out,err}` |
