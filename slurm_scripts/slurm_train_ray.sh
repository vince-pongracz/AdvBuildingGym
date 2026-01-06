#!/usr/bin/env bash
# Author: Vince Pongracz
# Maintainer: uchwd@student.kit.edu
# Created: 2026-01-02 | Version: 1.0
# Description: Submit a SLURM job that runs `run_train_ray.py` using Ray/RLLib

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_train_ray.sh [ALGORITHM] [TIMESTEPS] [SEED] [METRIC] [CONFIG_NAME]
#
# Examples:
#   sbatch slurm_scripts/slurm_train_ray.sh ppo 1000000 42 reward_rate
#   sbatch slurm_scripts/slurm_train_ray.sh ppo 1000000 42 achieved_reward test1
#
# Default values are set for all arguments so you can submit the job without
# positional arguments. The script activates the project's Python virtualenv
# (relative path) and runs the training script while logging SLURM and GPU info.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:full:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs_train/slurm-train-ray-%j.out
#SBATCH --error=slurm_logs_train/slurm-train-ray-%j.err
#SBATCH --job-name=ray-train-%j

set -euo pipefail

# Activate virtual environment (adjust path if your env is located elsewhere)
PYTHON_ENV="../adv_env"
if [ -d "$PYTHON_ENV" ]; then
  # Prefer absolute path activation inside SLURM jobs
  source "${PYTHON_ENV}/bin/activate"
  echo "=== Python and pip versions ==="
  python --version
  pip --version
else
  echo "[WARN] Python environment not found at ${PYTHON_ENV}; continuing without activation"
fi

# Parse positional arguments (defaults mirror run_train_ray.py)
ALGORITHM=${1:-ppo}
TIMESTEPS=${2:-1000000}
SEED=${3:-42}
METRIC=${4:-reward_rate}
CONFIG_NAME=${5:-}

echo "=== Starting Ray training job ==="
echo "  Algorithm : $ALGORITHM"
echo "  Timesteps : $TIMESTEPS"
echo "  Seed      : $SEED"
echo "  Metric    : $METRIC"
if [ -n "$CONFIG_NAME" ]; then
  echo "  Config    : $CONFIG_NAME"
fi

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-}"

# Unset LD_LIBRARY_PATH to avoid conflicts at torch / cuDNN (cuDNN mismatch occoured in earlier tests)
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

# Disable ANSI color codes and log deduplication in Ray logs
export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=0
export TERM=dumb
# Force unbuffered Python output for immediate log visibility
export PYTHONUNBUFFERED=1

# Build command
CMD=(python -u run_train_ray.py --algorithm "$ALGORITHM" --timesteps "$TIMESTEPS" --seed "$SEED" --metric "$METRIC" --training)
if [ -n "$CONFIG_NAME" ]; then
  CMD+=(--config_name "$CONFIG_NAME")
fi

echo "======"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Training completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_train_ray.sh
# - Submit with positional args:
#     sbatch slurm_scripts/slurm_train_ray.sh [ALGORITHM] [TIMESTEPS] [SEED] [METRIC] [CONFIG_NAME]
# - Available metrics: episode_return_mean, achieved_reward, reward_rate
# - Output and error logs will be written to `slurm_logs_train/`.
# -------------------------------------------------------------------------------
