# SLURM Scripts

SLURM quickstart: https://slurm.schedmd.com/quickstart.html
HaiCORE batch docs: https://www.nhr.kit.edu/userdocs/haicore/batch/

All scripts activate the Python virtualenv at `../adv_env` relative to the project root.

## Scripts overview

| Script | Purpose | GPU | CPUs | Default time |
|--------|---------|-----|------|--------------|
| `slurm_train_ray.sh` | Ray/RLlib training | 1x full | 4 | 10 min |
| `slurm_eval_ray.sh` | Ray/RLlib evaluation | No | 2 | 10 min |
| `slurm_train_sb.sh` | Stable Baselines3 training | 1x full | 4 | 15 min |
| `slurm_data_setup.sh` | Data fetching and preprocessing | No | 2 | 10 min |
| `slurm_plot_trajectory.sh` | Trajectory plotting from HDF5 | No | 1 | 10 min |
| `slurm_check_gpu_info.sh` | GPU diagnostics | 1x 4g.20gb | - | 5 min |

## Log directories

Each script writes logs to a dedicated subdirectory under `slurm_logs/`:

| Script | Log directory |
|--------|--------------|
| `slurm_train_ray.sh` | `slurm_logs/train/` |
| `slurm_eval_ray.sh` | `slurm_logs/eval/` |
| `slurm_train_sb.sh` | `slurm_logs/train/` |
| `slurm_data_setup.sh` | `slurm_logs/data_setup/` |
| `slurm_plot_trajectory.sh` | `slurm_logs/plotting/` |
| `slurm_check_gpu_info.sh` | `slurm_logs/train/` |

Log files are named `slurm-<type>-<jobid>.out` / `.err`. Higher job ID = more recent run.
The `.err` files are the primary source for troubleshooting.

## Overriding SLURM parameters from the command line

Any `#SBATCH` directive in a script can be overridden by passing the same flag
to `sbatch` on the command line. CLI flags always take precedence over directives
in the script. This is a native SLURM feature -- no wrapper or env var needed.

The most common override is `--time` for short vs long runs:

```bash
# Short run (uses the 10 min default in the script)
sbatch slurm_scripts/slurm_train_ray.sh --algorithm sac --episodes 500

# Long run (override time to 4 hours)
sbatch --time=04:00:00 slurm_scripts/slurm_train_ray.sh --algorithm sac --episodes 5000

# Override multiple SLURM params at once
sbatch --time=02:00:00 --cpus-per-task=8 slurm_scripts/slurm_train_ray.sh --algorithm ppo
```

Other useful overrides:

```bash
# Request more CPUs for parallel env runners
sbatch --cpus-per-task=8 slurm_scripts/slurm_train_ray.sh --algorithm ppo

# Run on a specific partition
sbatch --partition=dev_gpu_4 slurm_scripts/slurm_train_ray.sh --algorithm sac

# Exclude problematic nodes (in addition to those already excluded in the script)
sbatch --exclude=haicn1704,haicn1711,haicn1720 slurm_scripts/slurm_train_ray.sh
```

## Ray/RLlib training (slurm_train_ray.sh)

Primary training script. All arguments after the script name are forwarded to
`run_train_ray.py`. The `--training` flag is always injected automatically.

```bash
sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --episodes 3500 --seed 42
sbatch slurm_scripts/slurm_train_ray.sh --algorithm sac --load-config configs/my_config.yaml
sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --episodes 5000 --checkpoint-frequency-episodes 50
```

Key `run_train_ray.py` options:

| Flag | Description | Default |
|------|-------------|---------|
| `--algorithm` | `ppo` or `sac` | `ppo` |
| `--episodes N` | Total training episodes | 3500 |
| `-cn NAME` | Configuration name | `env_test1_{s/m/l}` |
| `--seed N` | Random seed | 42 |
| `--metric METRIC` | Optimisation target (`reward_rate`, `achieved_reward`, `episode_return_mean`) | `reward_rate` |
| `--checkpoint-frequency-episodes N` | Checkpoint every N episodes | 20 |
| `--load-config PATH` | Load config from YAML | - |
| `--save-config PATH` | Save config to YAML | - |
| `--log-trajectories` | Save per-step trajectory JSON during eval | off |

The script also sets up CUDA environment variables and fixes cuDNN library
paths for the HaiCORE cluster.

## Ray/RLlib evaluation (slurm_eval_ray.sh)

Runs CPU-only inference (no GPU needed for the small `[32,32,32]` network).
All arguments are forwarded to `run_eval_ray.py`.

```bash
sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --episodes 10 --seed 42
sbatch slurm_scripts/slurm_eval_ray.sh --algorithm sac -cn env_test1_{s/m/l} --episodes 20
sbatch slurm_scripts/slurm_eval_ray.sh --checkpoint models/env_test1_{s/m/l}/ray/ppo/best_model_ep100
```

Key `run_eval_ray.py` options:

| Flag | Description | Default |
|------|-------------|---------|
| `--algorithm, -a` | `ppo` or `sac` | `ppo` |
| `--checkpoint PATH` | Checkpoint directory (auto-detects best if omitted) | - |
| `-cn NAME` | Configuration name | - |
| `--episodes N` | Number of eval episodes | 10 |
| `--seed N` | Random seed | 42 |
| `--output-dir PATH` | Results directory | `eval_results` |
| `--log-trajectories` | Save per-step trajectory JSON (on by default) | on |
| `--no-save` | Skip saving results to disk | - |

## Stable Baselines3 training (slurm_train_sb.sh)

Secondary training script using Stable Baselines3. Uses positional arguments
(not named flags like the Ray scripts).

```bash
sbatch slurm_scripts/slurm_train_sb.sh                              # all defaults
sbatch slurm_scripts/slurm_train_sb.sh ppo 4 1000000 42 my_config   # positional args
```

Positional arguments: `[ALGORITHM] [NUM_ENVS] [TIMESTEPS] [SEED] [CONFIG_NAME]`

Defaults: `ppo`, `4` envs, `1000000` timesteps, seed `42`.

## Data setup (slurm_data_setup.sh)

Fetches and preprocesses energy price and weather data. CPU-only.
All arguments are forwarded to `preproc/data_setup.py`.

```bash
sbatch slurm_scripts/slurm_data_setup.sh
sbatch slurm_scripts/slurm_data_setup.sh --skip-weather
sbatch slurm_scripts/slurm_data_setup.sh --skip-weather --skip-price-fetch --years 2023 --augment
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `--years YEAR [...]` | Target years for price data | 2017-2026 |
| `--price-source` | `awattar` or `energy-charts` | `awattar` |
| `--skip-prices` | Skip entire price pipeline | - |
| `--skip-price-fetch` | Skip fetching, use existing raw files | - |
| `--skip-weather` | Skip all weather pipelines (WPuQ/Zenodo and DWD) | - |
| `--skip-wpuq` | Skip WPuQ/Zenodo weather pipeline only | - |
| `--skip-dwd` | Skip DWD weather pipeline only | - |
| `--augment` | Run price augmentation after preprocessing | - |

## Trajectory plotting (slurm_plot_trajectory.sh)

Generates plots from HDF5 evaluation files. CPU-only.
All arguments are forwarded to `plotting.src.trajectory_plot`.

```bash
sbatch slurm_scripts/slurm_plot_trajectory.sh
sbatch slurm_scripts/slurm_plot_trajectory.sh --hdf5 eval_results/latest/trajectories/trajectories.hdf5
sbatch slurm_scripts/slurm_plot_trajectory.sh --episode ep_42 --format html svg png
sbatch slurm_scripts/slurm_plot_trajectory.sh --select-by achieved_reward
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `--hdf5 PATH` | Path to trajectories.hdf5 | auto-discover latest |
| `--episode ID` | Episode ID to plot | best by `--select-by` |
| `--format FMT [...]` | Output formats: `html`, `png`, `svg`, `pdf` | `html svg` |
| `--select-by METRIC` | Metric for best-episode selection | `reward_rate` |

## GPU diagnostics (slurm_check_gpu_info.sh)

Quick job to verify GPU availability and CUDA configuration on a cluster node.

```bash
sbatch slurm_scripts/slurm_check_gpu_info.sh
```

## Utility

`slurm_scripts/util/print_env_info.py` prints Python, PyTorch, and CUDA
environment info. Called automatically by the training and evaluation scripts
for consistent diagnostics in job logs.
