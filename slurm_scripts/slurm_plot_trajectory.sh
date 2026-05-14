#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-03-13 | Version: 1.0
# Description: Submit a SLURM job that generates trajectory plots from HDF5 evaluation files

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_plot_trajectory.sh [OPTIONS]
#
# All arguments are forwarded directly to plotting.traj_plotting.trajectory_plot.
# Available options:
#   --hdf5 PATH             Path to trajectories.hdf5 file (default: auto-discover latest)
#   --episode ID            Episode ID to plot (default: best by --select-by metric)
#   --all-episodes          Plot every episode in the HDF5 (one subdir per episode)
#   --output-dir PATH       Output directory (default: plotting/out/<episode_id>/)
#   --format FMT [FMT ...]  Output format(s): html, png, svg, pdf (default: html svg)
#   --control-step N        Control timestep in seconds (default: 300)
#   --select-by METRIC      Metric for best-episode selection: reward_rate, achieved_reward, cum_E_kWh
#
# Examples:
#   sbatch slurm_scripts/slurm_plot_trajectory.sh
#   sbatch slurm_scripts/slurm_plot_trajectory.sh --hdf5 ep_metrics/trajectories/trajectories.hdf5
#   sbatch slurm_scripts/slurm_plot_trajectory.sh --episode ep_42 --format html svg png
#   sbatch slurm_scripts/slurm_plot_trajectory.sh --all-episodes --hdf5 eval_results/run_X/trajectories.hdf5
#   sbatch slurm_scripts/slurm_plot_trajectory.sh --select-by achieved_reward
#
# Note: Plotting runs on CPU only. No GPU is requested.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/plotting/slurm-plot-trajectory-%j.out
#SBATCH --error=slurm_logs/plotting/slurm-plot-trajectory-%j.err
#SBATCH --job-name=plot-traj-%j

set -euo pipefail

# Activate virtual environment
PYTHON_ENV="../adv_env"
if [ -d "$PYTHON_ENV" ]; then
  source "${PYTHON_ENV}/bin/activate"
  echo "=== Python and pip versions ==="
  python --version
  pip --version
else
  echo "[WARN] Python environment not found at ${PYTHON_ENV}; continuing without activation"
fi

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                : $(hostname)"

export PYTHONUNBUFFERED=1

# Forward all arguments directly to the plotting module.
# stdout/stderr are captured by SLURM via the #SBATCH --output / --error directives.
echo "======"
echo "Running: python -u -m plotting.traj_plotting.trajectory_plot $*"
python -u -m plotting.traj_plotting.trajectory_plot "$@"

echo "Trajectory plotting completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_plot_trajectory.sh
# - Submit with named arguments:
#     sbatch slurm_scripts/slurm_plot_trajectory.sh --hdf5 path/to/trajectories.hdf5
# - Output and error logs will be written to `slurm_logs/plotting/`.
# -------------------------------------------------------------------------------
