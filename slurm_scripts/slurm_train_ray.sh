#!/usr/bin/env bash
# Author: Vince Pongracz
# Maintainer: uchwd@student.kit.edu
# Created: 2026-01-02 | Version: 1.1
# Description: Submit a SLURM job that runs `run_train_ray.py` using Ray/RLLib

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_train_ray.sh [OPTIONS]
#
# All arguments are forwarded directly to run_train_ray.py. Available options:
#   --algorithm ALGO          Algorithm to use (ppo, sac) [default: ppo]
#   --load-config PATH        Path to YAML env config file to load (REQUIRED)
#   --episodes N              Total training episodes [default: 3500]
#   --seed N                  Random seed
#   --eval-freq N             Evaluation frequency [default: 20000]
#   --metric METRIC           Metric to optimize (episode_return_mean, achieved_reward, reward_rate) [default: reward_rate]
#   --checkpoint-frequency-episodes N   Checkpoint frequency in episodes [default: 20]
#   --log-trajectories              Save per-step trajectory JSON during eval episodes [default: off]
#   --no-log-trajectories           Disable trajectory logging (default)
#
# Examples (--load-config is REQUIRED):
#   sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --load-config configs/env/env_test1_small.yaml --episodes 3500 --seed 42
#   sbatch slurm_scripts/slurm_train_ray.sh --algorithm sac --load-config configs/env/env_test1_mid.yaml
#   sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --load-config configs/env/env_test1_large.yaml --episodes 5000 --checkpoint-frequency-episodes 50
#
# The script activates the project's Python virtualenv and runs the training
# script while logging SLURM and GPU info.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:full:1
#SBATCH --time=00:10:00
# Exclude nodes with known GPU issues (add problematic nodes here)
#SBATCH --exclude=haicn1704,haicn1711
#SBATCH --output=slurm_logs/train/slurm-train-ray-%j.out
#SBATCH --error=slurm_logs/train/slurm-train-ray-%j.err
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

# All arguments are forwarded directly to run_train_ray.py which owns the
# defaults (algorithm, episodes, seed, metric, etc.) via argparse.
SCRIPT_ARGS=("$@")

echo "=== Starting Ray training job ==="
echo "  Args: ${SCRIPT_ARGS[*]:-(none, using run_train_ray.py defaults)}"

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-}"

# Debug: Show LD_LIBRARY_PATH before any modifications
echo "=== LD_LIBRARY_PATH (before) ==="
echo "${LD_LIBRARY_PATH:-<not set>}"

# Fix cuDNN version mismatch: pip nvidia-cudnn-cu12 ships cuDNN 9.10.2, but
# system CUDA 12.4 has cuDNN 9.5.1.  cuDNN lives in the nvidia pip package
# (nvidia/cudnn/lib), NOT in torch/lib.  Prepend both so the linker finds
# the pip-installed versions before the system ones.
# Do NOT remove system CUDA paths — they provide libcuda.so (driver stub) on HPC clusters.
NVIDIA_CUDNN_LIB=$(python -c "import nvidia.cudnn; import os; print(os.path.join(nvidia.cudnn.__path__[0], 'lib'))" 2>/dev/null || echo "")
PYTORCH_LIB=$(python -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null || echo "")
PREPEND=""
for p in "$NVIDIA_CUDNN_LIB" "$PYTORCH_LIB"; do
    [ -n "$p" ] && [ -d "$p" ] && PREPEND="${PREPEND:+${PREPEND}:}${p}"
done
if [ -n "$PREPEND" ]; then
    export LD_LIBRARY_PATH="${PREPEND}:${LD_LIBRARY_PATH:-}"
    echo "=== LD_LIBRARY_PATH (nvidia+pytorch prepended) ==="
    echo "${LD_LIBRARY_PATH}"
fi

# CUDA initialization workarounds for HPC clusters
export CUDA_MODULE_LOADING=LAZY          # Delay CUDA init, can help with driver issues
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # Consistent GPU ordering with nvidia-smi
export TORCH_CUDA_ARCH_LIST="8.0"        # A100 compute capability, skip auto-detection

echo "=== GPU Info (nvidia-smi) ==="
nvidia-smi || true

echo "=== Python / CUDA Info ==="
python slurm_scripts/util/print_env_info.py

# Disable ANSI color codes and log deduplication in Ray logs
export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=0
export TERM=dumb
# Force unbuffered Python output for immediate log visibility
export PYTHONUNBUFFERED=1

# Build command: Forward every param as they are
CMD=(python -u run_train_ray.py "${SCRIPT_ARGS[@]}")

echo "======"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Training completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_train_ray.sh
# - Submit with named arguments:
#     sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --episodes 3500 --seed 42
# - All arguments from run_train_ray.py are supported with their default values
# - Available metrics: episode_return_mean, achieved_reward, reward_rate
# - Output and error logs will be written to `slurm_logs/train/`.
# -------------------------------------------------------------------------------
