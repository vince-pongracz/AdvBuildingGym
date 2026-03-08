#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-03-05 | Version: 1.1
# Description: Submit a SLURM job that runs data setup and data fetching

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_data_setup.sh [OPTIONS]
#
# All arguments are forwarded directly to data_setup.py. Available options:
#   --years YEAR [YEAR ...]       Target years for price data (default: 2017-2026)
#   --price-source {awattar,energy-charts}  Price data source (default: awattar)
#   --skip-prices                 Skip the entire price pipeline
#   --skip-price-fetch            Skip fetching, use existing raw files
#   --skip-weather                Skip the entire weather/Zenodo pipeline
#   --steps STEP [STEP ...]       Select which steps to run
#   --augment                     Run price augmentation after preprocessing
#   --log-level {DEBUG,INFO,...}  Logging level (default: INFO)
#
# Examples:
#   sbatch slurm_scripts/slurm_data_setup.sh
#   sbatch slurm_scripts/slurm_data_setup.sh --skip-weather
#   sbatch slurm_scripts/slurm_data_setup.sh --skip-weather --skip-price-fetch --years 2023 --raw-price-files data/e_price/2023_prices.csv --augment
#
# Note: Data setup runs on CPU only. No GPU is requested.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/data_setup/slurm-data-setup-%j.out
#SBATCH --error=slurm_logs/data_setup/slurm-data-setup-%j.err
#SBATCH --job-name=data-setup-%j

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

echo "=== Python Info ==="
python slurm_scripts/util/print_env_info.py

# Disable ANSI color codes and log deduplication in Ray logs
export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=0
export TERM=dumb
export PYTHONUNBUFFERED=1
export RAY_SCHEDULER_EVENTS=0

# Forward all arguments directly to data_setup.py
CMD=(python -u preproc/data_setup.py "$@")

echo "======"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Data setup completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_data_setup.sh
# - Submit with named arguments:
#     sbatch slurm_scripts/slurm_data_setup.sh --skip-weather --years 2023 2024
# - Output and error logs will be written to `slurm_logs/data_setup/`.
# -------------------------------------------------------------------------------
