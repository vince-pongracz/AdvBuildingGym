#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-04-11 | Version: 1.1
# Description: SLURM job that runs data-visualisation plotting scripts (CPU only).
#
# Usage:
#   sbatch slurm_scripts/slurm_data_vis.sh              # run all plots in parallel
#   sbatch slurm_scripts/slurm_data_vis.sh cross_year   # run a single plot
#   sbatch slurm_scripts/slurm_data_vis.sh monthly
#
# Logs: slurm_logs/data_vis/slurm-data-vis-<jobid>.out  (combined stdout+stderr)
#       slurm_logs/data_vis/slurm-data-vis-<jobid>_<tag>.err (per-plot stderr)
#
# Future-date warnings ('No data for <date>') are OFF by default — the plot
# scripts default `--warn-future-data` to False, and this script forwards no
# extra args. To enable them, invoke the plot module directly with the flag.
#
# SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/data_vis/slurm-data-vis-%j.out
#SBATCH --job-name=data-vis-%j

set -euo pipefail

# --- Config ------------------------------------------------------------------
LOG_DIR="slurm_logs/data_vis"
PYTHON_ENV="../adv_env"

declare -A PLOTS=(
  [cross_year]="plotting.data_plotting.plot_cross_year_combined"
  [monthly]="plotting.data_plotting.plot_monthly_overview"
)

# --- Parse argument (optional -- prefix, empty = run all) --------------------
TARGET="${1:-}"
if [[ "${TARGET}" == --* ]]; then
  TARGET=""
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=("${@:2}")
  TARGET="${TARGET#--}"
fi
case "${TARGET}" in
  "")                 TAGS=("${!PLOTS[@]}") ;;
  cross_year|monthly) TAGS=("${TARGET}") ;;
  *) echo "[ERROR] Unknown target '${TARGET}' (expected: cross_year, monthly, or empty)"; exit 2 ;;
esac

# --- Environment -------------------------------------------------------------
mkdir -p "${LOG_DIR}"
LOG_PREFIX="${LOG_DIR}/slurm-data-vis-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

if [ -d "${PYTHON_ENV}" ]; then
  source "${PYTHON_ENV}/bin/activate"
  echo "=== Python: $(python --version 2>&1) | pip: $(pip --version)"
else
  echo "[WARN] Python env not found at ${PYTHON_ENV}; continuing without activation"
fi
echo "=== Node: $(hostname) | SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-?}"

# --- Launch ------------------------------------------------------------------
echo "=== Launching ${#TAGS[@]} plot(s) in parallel:"
declare -A PIDS=()
for tag in "${TAGS[@]}"; do
  echo "  - ${PLOTS[${tag}]}  (stderr -> ${LOG_PREFIX}_${tag}.err)"
  python -u -m "${PLOTS[${tag}]}" "${EXTRA_ARGS[@]}" 2>"${LOG_PREFIX}_${tag}.err" &
  PIDS[${tag}]=$!
done

# --- Wait and report ---------------------------------------------------------
STATUS=0
for tag in "${TAGS[@]}"; do
  rc=0; wait "${PIDS[${tag}]}" || rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "[ERROR] ${PLOTS[${tag}]} failed (exit ${rc}); see ${LOG_PREFIX}_${tag}.err"
    STATUS=${rc}
  fi
done

if [ "${STATUS}" -eq 0 ]; then
  echo "Data visualisation completed successfully."
else
  echo "Data visualisation completed with errors."
  exit "${STATUS}"
fi
