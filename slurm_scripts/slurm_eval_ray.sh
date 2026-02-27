#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-01-06 | Version: 1.2
# Description: Submit a SLURM job that evaluates a trained Ray/RLlib model

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_eval_ray.sh [OPTIONS]
#
# All arguments are forwarded directly to run_eval_ray.py. Available options:
#   --algorithm, -a ALGO    Algorithm to evaluate (ppo, sac, ddpg, td3, a2c) [default: ppo]
#   --config-name, -cn NAME Configuration name (used in checkpoint search path)
#   --load-config PATH      Path to JSON config file to load
#   --checkpoint PATH       Path to Ray checkpoint directory (auto-detects best if omitted)
#   --episodes N            Number of evaluation episodes [default: 10]
#   --seed N                Random seed [default: 42]
#   --output-dir PATH       Directory to save evaluation results [default: eval_results]
#   --no-save               Do not save results to file
#   --log-trajectories      Save per-step trajectory JSON per episode [default: on]
#   --no-log-trajectories   Disable trajectory logging
#
# Examples:
#   sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --episodes 10 --seed 42
#   sbatch slurm_scripts/slurm_eval_ray.sh --algorithm sac --config-name test1 --episodes 20
#   sbatch slurm_scripts/slurm_eval_ray.sh --checkpoint models/test1/ray/ppo/best_model_ep100 --episodes 5
#   sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --load-config configs/my_config.json
#
# Note: No GPU is requested. run_eval_ray.py runs inference on CPU (sufficient for
# the small [32,32,32] network) and falls back gracefully when no GPU is present.
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

# Default values (mirror run_eval_ray.py defaults)
ALGORITHM="ppo"
EPISODES="10"
SEED="42"
CONFIG_NAME=""
LOAD_CONFIG=""
CHECKPOINT=""
OUTPUT_DIR=""
NO_SAVE=0
LOG_TRAJECTORIES=""
EXTRA_ARGS=()

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --algorithm|-a)
      ALGORITHM="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --config-name|-cn)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --load-config)
      LOAD_CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --no-save)
      NO_SAVE=1
      shift
      ;;
    --log-trajectories)
      LOG_TRAJECTORIES="yes"
      shift
      ;;
    --no-log-trajectories)
      LOG_TRAJECTORIES="no"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "=== Starting Ray evaluation job ==="
echo "  Algorithm   : $ALGORITHM"
echo "  Episodes    : $EPISODES"
echo "  Seed        : $SEED"
[ -n "$CHECKPOINT" ]  && echo "  Checkpoint  : $CHECKPOINT"
[ -n "$CONFIG_NAME" ] && echo "  Config Name : $CONFIG_NAME"
[ -n "$LOAD_CONFIG" ] && echo "  Load Config : $LOAD_CONFIG"
[ -n "$OUTPUT_DIR" ]  && echo "  Output Dir  : $OUTPUT_DIR"
[ "$NO_SAVE" -eq 1 ]  && echo "  Save results: disabled"
[ "$LOG_TRAJECTORIES" = "yes" ] && echo "  Log Trajectories: enabled"
[ "$LOG_TRAJECTORIES" = "no" ]  && echo "  Log Trajectories: disabled"
[ ${#EXTRA_ARGS[@]} -gt 0 ] && echo "  Extra Args  : ${EXTRA_ARGS[*]}"

echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-}"
echo "Node                : $(hostname)"

echo "=== Python Info ==="
python slurm_scripts/util/print_env_info.py

# Disable ANSI color codes and log deduplication in Ray logs
export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=0
export TERM=dumb
# Force unbuffered Python output for immediate log visibility
export PYTHONUNBUFFERED=1

# Build command
CMD=(python -u run_eval_ray.py
  --algorithm "$ALGORITHM"
  --episodes "$EPISODES"
  --seed "$SEED"
)
[ -n "$CHECKPOINT" ]  && CMD+=(--checkpoint "$CHECKPOINT")
[ -n "$CONFIG_NAME" ] && CMD+=(-cn "$CONFIG_NAME")
[ -n "$LOAD_CONFIG" ] && CMD+=(--load-config "$LOAD_CONFIG")
[ -n "$OUTPUT_DIR" ]  && CMD+=(--output-dir "$OUTPUT_DIR")
[ "$NO_SAVE" -eq 1 ]  && CMD+=(--no-save)
[ "$LOG_TRAJECTORIES" = "yes" ] && CMD+=(--log-trajectories)
[ "$LOG_TRAJECTORIES" = "no" ]  && CMD+=(--no-log-trajectories)
[ ${#EXTRA_ARGS[@]} -gt 0 ] && CMD+=("${EXTRA_ARGS[@]}")

echo "======"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Evaluation completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_eval_ray.sh
# - Submit with named arguments:
#     sbatch slurm_scripts/slurm_eval_ray.sh --algorithm ppo --episodes 10 --seed 42
# - Output and error logs will be written to `slurm_logs_eval/`.
# -------------------------------------------------------------------------------
