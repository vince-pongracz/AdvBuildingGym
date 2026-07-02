#!/usr/bin/env bash
# Author: Vince Pongracz
# Description: Submit a SLURM job that evaluates a trained Stable-Baselines3 model

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_eval_sb.sh [OPTIONS]
#
# Forwarded directly to run_eval_sb.py.  Required:
#   --trial PATH            Path to trial config YAML
# Optional:
#   --checkpoint PATH       SB3 model .zip (extension optional), or 'best'/'latest'
#                           (auto-detects the best model under models/<trial>/sb3/<algo> if omitted)
#   --episodes N            Number of evaluation episodes [default: 10]
#   --output-dir PATH       Directory to save evaluation results [default: eval_results]
#   --no-save               Do not save results to file
#   --plot / --plot-all     Plot trajectory after evaluation
#   --stochastic            Sample from the policy instead of the deterministic action
#
# Examples:
#   sbatch slurm_scripts/slurm_eval_sb.sh --trial configs/trial_cfgs/trial_cfg_1_sac.yaml --episodes 10
#   sbatch slurm_scripts/slurm_eval_sb.sh --trial configs/trial_cfgs/trial_cfg_1_sac.yaml \
#         --checkpoint models/env_test1_small/sb3/sac/sac_seed42_.../best/best_model.zip
#
# Note: Inference runs on CPU (sufficient for the [256, 256] network). No GPU is requested.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/eval/slurm-eval-sb-%j.out
#SBATCH --error=slurm_logs/eval/slurm-eval-sb-%j.err
#SBATCH --job-name=eval-sb-%j

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

# Force unbuffered Python output for immediate log visibility.
export PYTHONUNBUFFERED=1

# Snapshot mode: when SNAPSHOT_DIR is set by tools/snapshot/submit_snapshot.py,
# extract the snapshot zip on demand, cd into the per-run dir, and run the
# snapshotted entry script instead of the live repo's copy.
ENTRY_SCRIPT_BASENAME="run_eval_sb.py"
# shellcheck source=util/snapshot_mode.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/snapshot_mode.sh"

# Start the per-minute scratch-disk usage sampler. Writes scratch_usage.log
# into the snapshot run dir (alongside slurm_<jobid>.{out,err}) in snapshot
# mode, or to ${SLURM_SUBMIT_DIR} in legacy mode.
# shellcheck source=util/scratch_monitor.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/scratch_monitor.sh"

# Forward all arguments directly to run_eval_sb.py
CMD=(python -u "${ENTRY_SCRIPT}" "$@")

echo "======"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Evaluation completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_eval_sb.sh
# - Submit:
#     sbatch slurm_scripts/slurm_eval_sb.sh --trial configs/trial_cfgs/trial_cfg_1_sac.yaml
# - Output and error logs will be written to `slurm_logs/eval/`.
# -------------------------------------------------------------------------------
