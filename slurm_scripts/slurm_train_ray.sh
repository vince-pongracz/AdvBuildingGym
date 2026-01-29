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
#   --algorithm ALGO          Algorithm to use (ppo, sac, ddpg, td3, a2c) [default: ppo]
#   --config_name, -cn NAME   Configuration name for the experiment
#   --load-config PATH        Path to JSON config file to load
#   --save-config PATH        Path to save config as JSON
#   --timesteps N             Total training timesteps [default: 1e6]
#   --num-envs N              Number of parallel environments [default: 1]
#   --seed N                  Random seed [default: 42]
#   --eval-freq N             Evaluation frequency [default: 20000]
#   --training                Use training split of price data (always enabled)
#   --metric METRIC           Metric to optimize (episode_return_mean, achieved_reward, reward_rate) [default: reward_rate]
#   --checkpoint-frequency-episodes N   Checkpoint frequency in episodes [default: 20]
#
# Examples:
#   sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --timesteps 1000000 --seed 42
#   sbatch slurm_scripts/slurm_train_ray.sh --algorithm sac --load-config configs/my_config.json
#   sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --checkpoint-frequency-episodes 50 --num-envs 4
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
#SBATCH --exclude=haicn1704
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

# Default values (mirror run_train_ray.py defaults)
ALGORITHM="ppo"
TIMESTEPS="1000000"
SEED="42"
METRIC="reward_rate"
CONFIG_NAME=""
LOAD_CONFIG=""
SAVE_CONFIG=""
NUM_ENVS="1"
EVAL_FREQ="20000"
CHECKPOINT_FREQ=""
EXTRA_ARGS=()

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --algorithm|-a)
      ALGORITHM="$2"
      shift 2
      ;;
    --timesteps)
      TIMESTEPS="$2"
      shift 2
      ;;
    --seed|-s)
      SEED="$2"
      shift 2
      ;;
    --metric)
      METRIC="$2"
      shift 2
      ;;
    --config_name|-cn)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --load-config)
      LOAD_CONFIG="$2"
      shift 2
      ;;
    --save-config)
      SAVE_CONFIG="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --eval-freq)
      EVAL_FREQ="$2"
      shift 2
      ;;
    --checkpoint-frequency-episodes)
      CHECKPOINT_FREQ="$2"
      shift 2
      ;;
    --training)
      # Ignored since we always add --training
      shift
      ;;
    *)
      # Pass through any unknown arguments
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "=== Starting Ray training job ==="
echo "  Algorithm   : $ALGORITHM"
echo "  Timesteps   : $TIMESTEPS"
echo "  Seed        : $SEED"
echo "  Metric      : $METRIC"
echo "  Num Envs    : $NUM_ENVS"
echo "  Eval Freq   : $EVAL_FREQ"
[ -n "$CONFIG_NAME" ] && echo "  Config Name : $CONFIG_NAME"
[ -n "$LOAD_CONFIG" ] && echo "  Load Config : $LOAD_CONFIG"
[ -n "$SAVE_CONFIG" ] && echo "  Save Config : $SAVE_CONFIG"
[ -n "$CHECKPOINT_FREQ" ] && echo "  Checkpoint Freq: $CHECKPOINT_FREQ episodes"
[ ${#EXTRA_ARGS[@]} -gt 0 ] && echo "  Extra Args  : ${EXTRA_ARGS[*]}"

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-}"

# Debug: Show LD_LIBRARY_PATH before any modifications
echo "=== LD_LIBRARY_PATH (before) ==="
echo "${LD_LIBRARY_PATH:-<not set>}"

# Fix cuDNN version mismatch: PyTorch bundles cuDNN 9.10.2, but system CUDA 12.4 has cuDNN 9.5.1.
# Solution: Remove the system CUDA toolkit path entirely, then prepend PyTorch's lib path.
# The NVIDIA driver (libcuda.so) should be in /usr/lib64, accessible without the toolkit path.
PYTORCH_LIB=$(python -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null || echo "")
if [ -n "$LD_LIBRARY_PATH" ]; then
    # Remove system CUDA paths (which contain conflicting cuDNN)
    FILTERED_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cuda/' | tr '\n' ':' | sed 's/:$//')
    # Prepend PyTorch's lib path (contains correct cuDNN)
    if [ -n "$PYTORCH_LIB" ] && [ -d "$PYTORCH_LIB" ]; then
        export LD_LIBRARY_PATH="${PYTORCH_LIB}:${FILTERED_PATH}"
    else
        export LD_LIBRARY_PATH="${FILTERED_PATH}"
    fi
    echo "=== LD_LIBRARY_PATH (cuda removed, pytorch prepended) ==="
    echo "${LD_LIBRARY_PATH:-<empty>}"
fi

# CUDA initialization workarounds for HPC clusters
export CUDA_MODULE_LOADING=LAZY          # Delay CUDA init, can help with driver issues
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # Consistent GPU ordering with nvidia-smi
export TORCH_CUDA_ARCH_LIST="8.0"        # A100 compute capability, skip auto-detection

echo "=== GPU Info (nvidia-smi) ==="
nvidia-smi || true

echo "=== Python / CUDA Info ==="
python - <<'PY'
import os, sys
print('Python executable :', sys.executable)
print('Python version    :', sys.version.splitlines()[0])

# Check CUDA env vars before importing torch
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>'))
print('CUDA_MODULE_LOADING :', os.environ.get('CUDA_MODULE_LOADING', '<not set>'))
ld_path = os.environ.get('LD_LIBRARY_PATH', '<not set>')
print('LD_LIBRARY_PATH     :', ld_path[:200] + '...' if len(ld_path) > 200 else ld_path)

import torch
print('PyTorch version     :', torch.__version__)
print('CUDA built with     :', torch.version.cuda)

# Try explicit CUDA initialization with better error reporting
try:
    if torch.cuda.is_available():
        print('CUDA available      : True')
        print('Device count        :', torch.cuda.device_count())
        # Try to actually initialize CUDA
        torch.cuda.init()
        print('CUDA init           : SUCCESS')
        print('Device name         :', torch.cuda.get_device_name(0))
        print('CUDNN version       :', torch.backends.cudnn.version())
    else:
        print('CUDA available      : False')
        print('CUDA init failed - torch.cuda.is_available() returned False')
except Exception as e:
    print(f'CUDA init FAILED    : {type(e).__name__}: {e}')
PY

# Disable ANSI color codes and log deduplication in Ray logs
export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=0
export TERM=dumb
# Force unbuffered Python output for immediate log visibility
export PYTHONUNBUFFERED=1

# Build command with all arguments
CMD=(python -u run_train_ray.py
  --algorithm "$ALGORITHM"
  --timesteps "$TIMESTEPS"
  --seed "$SEED"
  --metric "$METRIC"
  --num-envs "$NUM_ENVS"
  --eval-freq "$EVAL_FREQ"
  --training
)
[ -n "$CONFIG_NAME" ] && CMD+=(--config_name "$CONFIG_NAME")
[ -n "$LOAD_CONFIG" ] && CMD+=(--load-config "$LOAD_CONFIG")
[ -n "$SAVE_CONFIG" ] && CMD+=(--save-config "$SAVE_CONFIG")
[ -n "$CHECKPOINT_FREQ" ] && CMD+=(--checkpoint-frequency-episodes "$CHECKPOINT_FREQ")
# Append any extra/unknown arguments
[ ${#EXTRA_ARGS[@]} -gt 0 ] && CMD+=("${EXTRA_ARGS[@]}")

echo "======"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Training completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_train_ray.sh
# - Submit with named arguments:
#     sbatch slurm_scripts/slurm_train_ray.sh --algorithm ppo --timesteps 1e6 --seed 42
# - All arguments from run_train_ray.py are supported with their default values
# - Available metrics: episode_return_mean, achieved_reward, reward_rate
# - Output and error logs will be written to `slurm_logs_train/`.
# -------------------------------------------------------------------------------
