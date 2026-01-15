#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-01-06 | Version: 1.0
# Description: Submit a SLURM job that evaluates a trained Ray/RLlib model

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_eval_ray.sh [CHECKPOINT_PATH] [EPISODES] [SEED]
#
# Examples:
#   sbatch slurm_scripts/slurm_eval_ray.sh
#   sbatch slurm_scripts/slurm_eval_ray.sh models/test1/best_ppo/ray/ppo_seed42/seed42_18064_/checkpoint_000000 20 42
#
# Default values are set for all arguments so you can submit the job without
# positional arguments. The script activates the project's Python virtualenv
# and runs the evaluation script while logging SLURM info.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:full:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs_eval/slurm-eval-ray-%j.out
#SBATCH --error=slurm_logs_eval/slurm-eval-ray-%j.err
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

# Parse positional arguments (defaults use latest checkpoint)
CHECKPOINT=${1:-}
EPISODES=${2:-10}
SEED=${3:-42}

echo "=== Starting Ray evaluation job ==="
if [ -n "$CHECKPOINT" ]; then
  echo "  Checkpoint : $CHECKPOINT"
else
  echo "  Checkpoint : (using latest/default)"
fi
echo "  Episodes   : $EPISODES"
echo "  Seed       : $SEED"

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-}"

# Unset LD_LIBRARY_PATH to avoid conflicts at torch / cuDNN
unset LD_LIBRARY_PATH

echo "=== GPU Info (nvidia-smi) ==="
nvidia-smi || true

echo "=== Python / CUDA Info ==="
python - <<'PY'
import torch, sys
print('Python executable :', sys.executable)
print('Python version    :', sys.version.splitlines()[0])
print('CUDA available    :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device name       :', torch.cuda.get_device_name(0))
    print('CUDA version (torch):', torch.version.cuda)
    print('CUDNN version     :', torch.backends.cudnn.version())
PY

# Build command
if [ -n "$CHECKPOINT" ]; then
  CMD=(python run_evaluation_ray.py --checkpoint "$CHECKPOINT" --episodes "$EPISODES" --seed "$SEED")
else
  CMD=(python run_evaluation_ray.py --episodes "$EPISODES" --seed "$SEED")
fi

echo "======"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Evaluation completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_eval_ray.sh
# - Submit with positional args:
#     sbatch slurm_scripts/slurm_eval_ray.sh [CHECKPOINT_PATH] [EPISODES] [SEED]
# - Output and error logs will be written to `slurm_logs_eval/`.
# -------------------------------------------------------------------------------
