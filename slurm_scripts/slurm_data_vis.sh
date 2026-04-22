#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-04-11 | Version: 1.0
# Description: Submit a SLURM job that runs plot_cross_year_combined and
#              plot_monthly_overview in parallel.

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_data_vis.sh
#
# Note: Data visualisation runs on CPU only. No GPU is requested.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/data_vis/slurm-data-vis-%j.out
#SBATCH --job-name=data-vis-%j

# stderr is merged into --output (no separate top-level .err file);
# per-script stderr goes to slurm-data-vis-<jobid>_{cross_year,monthly}.err

set -euo pipefail

mkdir -p slurm_logs/data_vis

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

LOG_DIR="slurm_logs/data_vis"
LOG_PREFIX="${LOG_DIR}/slurm-data-vis-${SLURM_JOB_ID}"

echo "======"
echo "Launching plot_cross_year_combined and plot_monthly_overview in parallel"
echo "Per-script logs:"
echo " ${LOG_PREFIX}_cross_year.err"
echo " ${LOG_PREFIX}_monthly.err"
echo "======"

# SLURM manages CPU affinity via cgroups — do not use taskset, as the
# allocated core IDs are not guaranteed to be 0, 1, etc.
python -u -m plotting.data_plotting.plot_cross_year_combined \
  2>"${LOG_PREFIX}_cross_year.err" &
PID_CROSS_YEAR=$!

python -u -m plotting.data_plotting.plot_monthly_overview \
  2>"${LOG_PREFIX}_monthly.err" &
PID_MONTHLY=$!

STATUS=0
wait "${PID_CROSS_YEAR}" || STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
  echo "[ERROR] plot_cross_year_combined failed with exit code ${STATUS}"
  echo "        See ${LOG_PREFIX}_cross_year.err"
fi

wait "${PID_MONTHLY}" || MONTHLY_STATUS=$?
MONTHLY_STATUS=${MONTHLY_STATUS:-0}
if [ "${MONTHLY_STATUS}" -ne 0 ]; then
  echo "[ERROR] plot_monthly_overview failed with exit code ${MONTHLY_STATUS}"
  echo "        See ${LOG_PREFIX}_monthly.err"
  STATUS=${MONTHLY_STATUS}
fi

if [ "${STATUS}" -eq 0 ]; then
  echo "Data visualisation completed successfully."
else
  echo "Data visualisation completed with errors."
  exit "${STATUS}"
fi

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_data_vis.sh
# - Submit:
#     sbatch slurm_scripts/slurm_data_vis.sh
# - Output and error logs will be written to `slurm_logs/data_vis/`.
# -------------------------------------------------------------------------------
