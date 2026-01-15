#!/usr/bin/env bash
# Author: Vince Pongracz
# Maintainer: uchwd@student.kit.edu
# Created: 2026-01-03 | Version: 1.0
# Description: Submit a SLURM job that runs `run_train_sb.py` using Stable Baselines3

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_script/slurm_train_sb.sh [ALGORITHM] [NUM_ENVS] [TIMESTEPS] [SEED] [CONFIG_NAME]
#
# Example:
#   sbatch slurm_script/slurm_train_sb.sh ppo 4 1000000 42 default_config
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
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs_train/slurm-train-sb-%j.out
#SBATCH --error=slurm_logs_train/slurm-train-sb-%j.err
#SBATCH --job-name=sb-train-%j

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

# Parse positional arguments (defaults mirror run_train_sb.py)
ALGORITHM=${1:-ppo}
NUM_ENVS=${2:-4}
TIMESTEPS=${3:-1000000}
SEED=${4:-42}
CONFIG_NAME=${5:-}

echo "=== Starting Stable Baselines3 training job ==="
echo "  Algorithm : $ALGORITHM"
echo "  Num envs  : $NUM_ENVS"
echo "  Timesteps : $TIMESTEPS"
echo "  Seed      : $SEED"
if [ -n "$CONFIG_NAME" ]; then
  echo "  Config    : $CONFIG_NAME"
fi

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-}"

# Unset LD_LIBRARY_PATH to avoid conflicts at torch / cuDNN (cuDNN mismatch occurred in earlier tests)
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
CMD=(python run_train_sb.py --algorithm "$ALGORITHM" --num-envs "$NUM_ENVS" --timesteps "$TIMESTEPS" --seed "$SEED")
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
#     chmod +x slurm_script/slurm_train_sb.sh
# - Submit with:
#     sbatch slurm_script/slurm_train_sb.sh [ALGORITHM] [NUM_ENVS] [TIMESTEPS] [SEED] [CONFIG_NAME]
# - Output and error logs will be written to `slurm_logs_train/`.
# -------------------------------------------------------------------------------
