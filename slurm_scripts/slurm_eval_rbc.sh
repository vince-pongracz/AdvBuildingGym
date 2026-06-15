#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-06-15 | Version: 1.0
# Description: Submit a SLURM job that evaluates rule-based control strategies

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_eval_rbc.sh [OPTIONS]
#
# Forwarded directly to run_eval_rule_based.py.  Required:
#   --trial PATH            Path to trial config YAML
# Optional:
#   --strategy NAME         Strategy to evaluate, or "all" [default: all]
#   --episodes N            Number of evaluation episodes per strategy [default: 10]
#   --seed N                Base seed (episode N uses seed + (N-1)); defaults to trial seed
#   --output-dir PATH       Directory to save evaluation results [default: eval_results]
#   --no-save               Do not save results to file
#   --evening-start H       Evening discharge window start hour (self_coverage) [default: 17.0]
#   --evening-end H         Evening discharge window end hour (self_coverage) [default: 23.0]
#   --preserve-start-soc    Forbid ending below the start-of-episode battery SoC [default: on]
#   --plot / --plot-all     Plot trajectory after evaluation
#
# Examples:
#   sbatch slurm_scripts/slurm_eval_rbc.sh --trial configs/trial_cfgs/trial_cfg_1.yaml --episodes 10
#   sbatch slurm_scripts/slurm_eval_rbc.sh --trial configs/trial_cfgs/trial_cfg_1.yaml \
#         --strategy price_median
#
# Note: Rule-based eval is CPU-only (no NN inference). No GPU is requested.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/eval/slurm-eval-rbc-%j.out
#SBATCH --error=slurm_logs/eval/slurm-eval-rbc-%j.err
#SBATCH --job-name=eval-rbc-%j

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
python "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/print_env_info.py"

# Disable ANSI color codes and log deduplication in Ray logs
# TODO noprio VP: Ray is not used in RBC eval...
export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=0
export RAY_SCHEDULER_EVENTS=0
export TERM=dumb
export PYTHONUNBUFFERED=1

# Snapshot mode: when SNAPSHOT_DIR is set by tools/snapshot/submit_snapshot.py,
# extract the snapshot zip on demand, cd into the per-run dir, and run the
# snapshotted entry script instead of the live repo's copy.
ENTRY_SCRIPT_BASENAME="run_eval_rule_based.py"
# shellcheck source=util/snapshot_mode.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/snapshot_mode.sh"

# Start the per-minute scratch-disk usage sampler. Writes scratch_usage.log
# into the snapshot run dir (alongside slurm_<jobid>.{out,err}) in snapshot
# mode, or to ${SLURM_SUBMIT_DIR} in legacy mode.
# shellcheck source=util/scratch_monitor.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/scratch_monitor.sh"

# Forward all arguments directly to run_eval_rule_based.py
CMD=(python -u "${ENTRY_SCRIPT}" "$@")

echo "======"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Evaluation completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_eval_rbc.sh
# - Submit:
#     sbatch slurm_scripts/slurm_eval_rbc.sh --trial configs/trial_cfgs/trial_cfg_1.yaml
# - Output and error logs will be written to `slurm_logs/eval/`.
# -------------------------------------------------------------------------------
