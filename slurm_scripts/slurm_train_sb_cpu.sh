#!/usr/bin/env bash
# Author: Vince Pongracz
# Maintainer: uchwd@student.kit.edu
# Created: 2026-05-28 | Version: 1.0
# Description: Submit a SLURM job that runs `run_train_sb.py` (Stable-Baselines3)
#              on CPU only — no GPU is requested, --cpu is forced.

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_train_sb_cpu.sh [OPTIONS]
#
# Forwarded directly to run_train_sb.py. Single required option:
#   --trial PATH      Path to trial config YAML (REQUIRED)
#
# --cpu is hard-coded by this wrapper, so the entry script always picks the
# CPU-only execution path. All run parameters (algorithm, seed, episodes,
# metric, checkpoint cadence, num_envs, schedules) live inside the trial YAML.
#
# Examples:
#   sbatch slurm_scripts/slurm_train_sb_cpu.sh --trial configs/trial_cfgs/trial_cfg_1_sac.yaml
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
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/train/slurm-train-sb-cpu-%j.out
#SBATCH --error=slurm_logs/train/slurm-train-sb-cpu-%j.err
#SBATCH --job-name=sb-train-cpu-%j

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

# Snapshot mode: when SNAPSHOT_DIR is set by tools/snapshot/submit_snapshot.py,
# extract the snapshot zip on demand, cd into the per-run dir, and run the
# snapshotted entry script instead of the live repo's copy.
ENTRY_SCRIPT_BASENAME="run_train_sb.py"
# shellcheck source=util/snapshot_mode.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/snapshot_mode.sh"

# Start the per-minute scratch-disk usage sampler. Writes scratch_usage.log
# into the snapshot run dir (alongside slurm_<jobid>.{out,err}) in snapshot
# mode, or to ${SLURM_SUBMIT_DIR} in legacy mode.
# shellcheck source=util/scratch_monitor.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/scratch_monitor.sh"

# All arguments are forwarded directly to run_train_sb.py which owns the
# CLI (--trial, --cpu) — defaults live in the trial YAML.
SCRIPT_ARGS=("$@")

echo "=== Starting Stable-Baselines3 CPU training job ==="
echo "  Args: ${SCRIPT_ARGS[*]:-(none, run_train_sb.py will fail without --trial)}"

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK  : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                 : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-}"

echo "=== Python / CUDA Info ==="
python "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/print_env_info.py"

# Force unbuffered Python output for immediate log visibility.
export PYTHONUNBUFFERED=1

# Build command: --cpu is hard-coded here to pin this wrapper to the CPU-only
# execution path regardless of the entry script. ENTRY_SCRIPT is either
# run_train_sb.py (legacy live-repo mode) or an absolute path inside the
# snapshot's code/ dir (snapshot mode).
CMD=(python -u "${ENTRY_SCRIPT}" --cpu "${SCRIPT_ARGS[@]}")

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
#     chmod +x slurm_scripts/slurm_train_sb_cpu.sh
# - Submit (CPU-only):
#     sbatch slurm_scripts/slurm_train_sb_cpu.sh --trial configs/trial_cfgs/trial_cfg_1_sac.yaml
# - For trials with num_envs > 3, pass extra CPUs:
#     sbatch --cpus-per-task=8 slurm_scripts/slurm_train_sb_cpu.sh --trial <path>
# - Output and error logs are written to slurm_logs/train/.
# -------------------------------------------------------------------------------
