#!/usr/bin/env bash
# Author: Vince Pongracz
# Maintainer: uchwd@student.kit.edu
# Created: 2026-01-03 | Version: 2.0
# Description: Submit a SLURM job that runs `run_train_sb.py` (Stable-Baselines3)

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_train_sb.sh [OPTIONS]
#
# Forwarded directly to run_train_sb.py. Single required option:
#   --trial PATH      Path to trial config YAML (REQUIRED)
#   --cpu             Optional: skip the GPU requirement (CPU smoke test);
#                     useful when allocating a CPU-only SLURM job.
#
# All run parameters (algorithm, seed, episodes, metric, checkpoint cadence,
# num_envs, schedules) live inside the trial YAML — see configs/trial_cfgs/*.yaml.
#
# Examples:
#   sbatch slurm_scripts/slurm_train_sb.sh --trial configs/trial_cfgs/trial_cfg_1_sac.yaml
#   sbatch --time=02:00:00 slurm_scripts/slurm_train_sb.sh --trial configs/trial_cfgs/trial_cfg_1_ppo.yaml
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# SB3 runs the learner in the driver process — no separate Ray actors —
# so we only need: 1 driver/learner CPU + 1 CPU per SubprocVecEnv worker
# + 1 small eval (in-process DummyVecEnv). 4 CPUs covers num_envs=1..3
# comfortably; bump --cpus-per-task on the sbatch line for larger num_envs.
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/train/slurm-train-sb-%j.out
#SBATCH --error=slurm_logs/train/slurm-train-sb-%j.err
#SBATCH --job-name=sb-train-%j

set -euo pipefail

# Activate the project's Python virtualenv.
PYTHON_ENV="../adv_env"
if [ -d "$PYTHON_ENV" ]; then
  source "${PYTHON_ENV}/bin/activate"
  echo "=== Python and pip versions ==="
  python --version
  pip --version
else
  echo "[WARN] Python environment not found at ${PYTHON_ENV}; continuing without activation"
fi

# All arguments are forwarded directly to run_train_sb.py which owns the
# CLI (--trial, --cpu) — defaults live in the trial YAML.
SCRIPT_ARGS=("$@")

echo "=== Starting Stable-Baselines3 training job ==="
echo "  Args: ${SCRIPT_ARGS[*]:-(none, run_train_sb.py will fail without --trial)}"

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK  : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                 : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-}"

# Fix cuDNN version mismatch on the cluster: pip nvidia-cudnn-cu12 ships
# cuDNN 9.10.2 but system CUDA 12.4 has 9.5.1. Prepend the pip-installed
# cuDNN + torch lib dirs so the linker finds them first. Do NOT remove
# system CUDA paths — they provide libcuda.so (driver stub). Same fix as
# slurm_train_ray.sh.
echo "=== LD_LIBRARY_PATH (before) ==="
echo "${LD_LIBRARY_PATH:-<not set>}"

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

# CUDA initialization workarounds for HPC clusters.
export CUDA_MODULE_LOADING=LAZY          # Delay CUDA init, helps with driver flakiness
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # Consistent GPU ordering with nvidia-smi
export TORCH_CUDA_ARCH_LIST="8.0"        # A100 compute capability, skip auto-detection

echo "=== GPU Info (nvidia-smi) ==="
nvidia-smi || true

echo "=== Python / CUDA Info ==="
python "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/print_env_info.py"

# Force unbuffered Python output for immediate log visibility.
export PYTHONUNBUFFERED=1

# Snapshot mode: when SNAPSHOT_DIR is set by tools/snapshot/submit_snapshot.py,
# extract the snapshot zip on demand, cd into the per-run dir, and run the
# snapshotted entry script instead of the live repo's copy.
ENTRY_SCRIPT_BASENAME="run_train_sb.py"
# shellcheck source=util/snapshot_mode.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/snapshot_mode.sh"

# Start the per-minute scratch-disk usage sampler. Writes scratch_usage.log
# next to slurm.err in snapshot mode, or to ${SLURM_SUBMIT_DIR} in legacy mode.
# shellcheck source=util/scratch_monitor.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/scratch_monitor.sh"

CMD=(python -u "${ENTRY_SCRIPT}" "${SCRIPT_ARGS[@]}")

echo "======"
echo "Running: ${CMD[*]}"

set +e
"${CMD[@]}"
TRAIN_EC=$?
set -e

if [ "${TRAIN_EC}" -ne 0 ]; then
  echo "Training failed with exit code ${TRAIN_EC}"
  exit "${TRAIN_EC}"
fi

echo "Training completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_train_sb.sh
# - Submit (GPU):
#     sbatch slurm_scripts/slurm_train_sb.sh --trial configs/trial_cfgs/trial_cfg_1_sac.yaml
# - For trials with num_envs > 3, pass extra CPUs:
#     sbatch --cpus-per-task=8 slurm_scripts/slurm_train_sb.sh --trial <path>
# - For a CPU-only smoke test (no GPU), strip the gres line at submission:
#     sbatch --gres=NONE slurm_scripts/slurm_train_sb.sh --cpu --trial <path>
# - Output and error logs are written to slurm_logs/train/.
# -------------------------------------------------------------------------------
