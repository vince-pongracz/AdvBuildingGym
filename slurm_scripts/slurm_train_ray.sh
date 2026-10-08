#!/usr/bin/env bash
# Author: Vince Pongracz
# Maintainer: uchwd@student.kit.edu
# Created: 2026-01-02 | Version: 1.1
# Description: Submit a SLURM job that runs `run_train_ray.py` using Ray/RLLib

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_train_ray.sh [OPTIONS]
#
# Forwarded directly to run_train_ray.py. Single required option:
#   --trial PATH              Path to trial config YAML (REQUIRED)
#
# All run parameters (algorithm, seed, episodes, metric, checkpoint cadence,
# schedules) live inside the trial YAML — see configs/trial_cfgs/*.yaml.
#
# Examples:
#   sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1.yaml
#
# The script activates the project's Python virtualenv and runs the training
# script while logging SLURM and GPU info.
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# TODO VP: set to 32, 38, 16 later -- but adapt Ray to use all possible cpu cores available
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --time=00:30:00
# Exclude nodes with known GPU issues (add problematic nodes here)
# // # --exclude=haicn1704,haicn1711
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

# Snapshot mode: when SNAPSHOT_DIR is set by tools/snapshot/submit_snapshot.py,
# extract the snapshot zip on demand, cd into the per-run dir, and run the
# snapshotted entry script instead of the live repo's copy. See the shared
# helper for the full setup.
ENTRY_SCRIPT_BASENAME="run_train_ray.py"
# shellcheck source=util/snapshot_mode.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/snapshot_mode.sh"

# Start the per-minute scratch-disk usage sampler. Writes scratch_usage.log
# into the snapshot run dir (alongside slurm_<jobid>.{out,err}) in snapshot
# mode, or to ${SLURM_SUBMIT_DIR} in legacy mode. Helper disowns itself so
# the `wait` later in this script does not block on the sampling loop.
# shellcheck source=util/scratch_monitor.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/scratch_monitor.sh"

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
python "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/print_env_info.py"

# Disable ANSI color codes and log deduplication in Ray logs
export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=0
export TERM=dumb
# Force unbuffered Python output for immediate log visibility
export PYTHONUNBUFFERED=1

# Build command: Forward every param as they are. ENTRY_SCRIPT is either
# run_train_ray.py (legacy live-repo mode) or an absolute path inside the
# snapshot's code/ dir (snapshot mode).
CMD=(python -u "${ENTRY_SCRIPT}" "${SCRIPT_ARGS[@]}")

# Filter for harmless EnvRunner.__del__/sigterm_handler tracebacks Ray prints
# when env-runner actors are SIGTERM'd at the end of tuner.fit().  Tune kills
# those actors as soon as fit() returns, so the in-process ray.shutdown() call
# in run_train_ray.py cannot prevent the noise — we strip it at the stderr
# boundary instead.  See: slurm_scripts/util/filter_ray_shutdown_spam.awk
SHUTDOWN_SPAM_FILTER="${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/filter_ray_shutdown_spam.awk"

echo "======"
echo "Running: ${CMD[*]}"

set +e
"${CMD[@]}" 2> >(exec awk -f "${SHUTDOWN_SPAM_FILTER}" >&2)
TRAIN_EC=$?
set -e

# Drain the stderr filter's process substitution before this script exits so
# its buffered output is flushed into the SLURM .err file.
wait

if [ "${TRAIN_EC}" -ne 0 ]; then
    echo "Training failed with exit code ${TRAIN_EC}"
    exit "${TRAIN_EC}"
fi

echo "Training completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_train_ray.sh
# - Submit:
#     sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1.yaml
# - Output and error logs will be written to `slurm_logs/train/`.
# -------------------------------------------------------------------------------
