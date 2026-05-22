#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-01-06 | Version: 1.3
# Description: Submit a SLURM job that evaluates a trained Ray/RLlib model

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_eval_ray.sh [OPTIONS]
#
# Forwarded directly to run_eval_ray.py.  Required:
#   --trial PATH            Path to trial config YAML
# Optional:
#   --checkpoint PATH       Path to Ray checkpoint directory (auto-detects best if omitted)
#   --episodes N            Number of evaluation episodes [default: 10]
#   --output-dir PATH       Directory to save evaluation results [default: eval_results]
#   --no-save               Do not save results to file
#   --plot / --plot-all     Plot trajectory after evaluation
#
# Examples:
#   sbatch slurm_scripts/slurm_eval_ray.sh --trial configs/trial_cfgs/trial_cfg_1.yaml --episodes 10
#   sbatch slurm_scripts/slurm_eval_ray.sh --trial configs/trial_cfgs/trial_cfg_1.yaml \
#         --checkpoint models/env_test1_small/ray/ppo/best_model_ep100
#
# Note: Inference runs on CPU (sufficient for the small [32,32,32] network).
# No GPU is requested.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/eval/slurm-eval-ray-%j.out
#SBATCH --error=slurm_logs/eval/slurm-eval-ray-%j.err
#SBATCH --job-name=eval-ray-%j

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
export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=0
export TERM=dumb
export PYTHONUNBUFFERED=1
export RAY_SCHEDULER_EVENTS=0

# Snapshot mode: when SNAPSHOT_DIR is set by tools/snapshot/submit_snapshot.py,
# extract the snapshot zip on demand, cd into the per-run dir, and run the
# snapshotted entry script instead of the live repo's copy.
ENTRY_SCRIPT_BASENAME="run_eval_ray.py"
# shellcheck source=util/snapshot_mode.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/snapshot_mode.sh"

# Forward all arguments directly to run_eval_ray.py
CMD=(python -u "${ENTRY_SCRIPT}" "$@")

echo "======"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Evaluation completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_eval_ray.sh
# - Submit:
#     sbatch slurm_scripts/slurm_eval_ray.sh --trial configs/trial_cfgs/trial_cfg_1.yaml
# - Output and error logs will be written to `slurm_logs/eval/`.
# -------------------------------------------------------------------------------
