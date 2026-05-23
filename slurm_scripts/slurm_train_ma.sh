#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-05-21 | Version: 1.0
# Description: Submit a SLURM job that runs `rl_ma_train.py` (multi-agent Ray/RLlib).

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_train_ma.sh [OPTIONS]
#
# Forwarded directly to rl_ma_train.py. Single required option:
#   --trial PATH              Path to trial config YAML (REQUIRED)
#
# Sibling of slurm_train_ray.sh; only the entry script differs. See that
# script and SLURM_README.md for the full set of forwarded CLI flags.
#
# Examples:
#   sbatch slurm_scripts/slurm_train_ma.sh --trial configs/trial_cfgs/trial_cfg_1.yaml
# -----------------------------------------------------------------------------

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/train/slurm-train-ma-%j.out
#SBATCH --error=slurm_logs/train/slurm-train-ma-%j.err
#SBATCH --job-name=ma-train-%j

set -euo pipefail

PYTHON_ENV="../adv_env"
if [ -d "$PYTHON_ENV" ]; then
  source "${PYTHON_ENV}/bin/activate"
  echo "=== Python and pip versions ==="
  python --version
  pip --version
else
  echo "[WARN] Python environment not found at ${PYTHON_ENV}; continuing without activation"
fi

# Snapshot mode: when SNAPSHOT_DIR is set by tools/snapshot/submit_snapshot.py,
# extract the snapshot zip on demand, cd into the per-run dir, and run the
# snapshotted entry script instead of the live repo's copy.
ENTRY_SCRIPT_BASENAME="rl_ma_train.py"
# shellcheck source=util/snapshot_mode.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/snapshot_mode.sh"

# Start the per-minute scratch-disk usage sampler. Writes scratch_usage.log
# next to slurm.err in snapshot mode, or to ${SLURM_SUBMIT_DIR} in legacy mode.
# shellcheck source=util/scratch_monitor.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/scratch_monitor.sh"

SCRIPT_ARGS=("$@")

echo "=== Starting multi-agent Ray training job ==="
echo "  Args: ${SCRIPT_ARGS[*]:-(none, rl_ma_train.py will fail without --trial)}"

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK  : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                 : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-}"

# Same cuDNN-path fix-up as slurm_train_ray.sh — see that script for the why.
NVIDIA_CUDNN_LIB=$(python -c "import nvidia.cudnn; import os; print(os.path.join(nvidia.cudnn.__path__[0], 'lib'))" 2>/dev/null || echo "")
PYTORCH_LIB=$(python -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null || echo "")
PREPEND=""
for p in "$NVIDIA_CUDNN_LIB" "$PYTORCH_LIB"; do
  [ -n "$p" ] && [ -d "$p" ] && PREPEND="${PREPEND:+${PREPEND}:}${p}"
done
if [ -n "$PREPEND" ]; then
  export LD_LIBRARY_PATH="${PREPEND}:${LD_LIBRARY_PATH:-}"
fi

export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDA_ARCH_LIST="8.0"

echo "=== GPU Info (nvidia-smi) ==="
nvidia-smi || true

echo "=== Python / CUDA Info ==="
python "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/print_env_info.py"

export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=0
export TERM=dumb
export PYTHONUNBUFFERED=1

CMD=(python -u "${ENTRY_SCRIPT}" "${SCRIPT_ARGS[@]}")

SHUTDOWN_SPAM_FILTER="${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/filter_ray_shutdown_spam.awk"

echo "======"
echo "Running: ${CMD[*]}"

set +e
"${CMD[@]}" 2> >(exec awk -f "${SHUTDOWN_SPAM_FILTER}" >&2)
TRAIN_EC=$?
set -e
wait

if [ "${TRAIN_EC}" -ne 0 ]; then
  echo "Training failed with exit code ${TRAIN_EC}"
  exit "${TRAIN_EC}"
fi

echo "Training completed successfully."
