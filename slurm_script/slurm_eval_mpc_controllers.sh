#!/usr/bin/env bash
# Author: G. Demirel (demirelg), KIT
# Maintainer: gokhan.demirel@kit.edu
# Created: 2025-06-17 | Version: 1.0
# Description: Evaluation script for MPC and Perfect MPC controllers

#SBATCH --job-name=eval-mpc-%j
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs_eval/slurm-eval-mpc-%j.out
#SBATCH --error=slurm_logs_eval/slurm-eval-mpc-%j.err

# Evaluation: MPC Control and Perfect MPC Control

set -euo pipefail

PYTHON_ENV="../llec_env"
source "${PYTHON_ENV}/bin/activate"

PYTHON_SCRIPT="run_evaluation.py"
OUTDOOR_PATH="data/LLEC_outdoor_temperature_5min_data.csv"

EPISODES=10
SEED=58
MPC_HORIZON=12

ALGORITHMS=(
  "Perfect MPC Control"
  "MPC Control"
)

# Valid (obs_variant => reward_mode) pairs
declare -A VALID_PAIRS=(
  ["T01"]="temperature"
  ["C01"]="combined"
)

for OBS in "${!VALID_PAIRS[@]}"; do
  REWARD_MODE="${VALID_PAIRS[$OBS]}"
  for ALG in "${ALGORITHMS[@]}"; do
    echo ">>> $ALG | $REWARD_MODE | $OBS"
    CMD="python $PYTHON_SCRIPT --algorithms \"$ALG\" --reward_mode $REWARD_MODE --obs_variant $OBS --episodes $EPISODES --seed $SEED --mpc_horizon $MPC_HORIZON --outdoor_temperature_path \"$OUTDOOR_PATH\""
    echo "[CMD] $CMD"
    eval $CMD
    echo "Done: $ALG | $REWARD_MODE | $OBS"
  done
done

echo "MPC evaluations completed."

# -------------------------------------------------------------------------------
# Notes:
# - Make sure the script is executable:
#     chmod +x slurm_script/slurm_eval_mpc_controllers.sh
#
# - To submit the job to SLURM:
#     sbatch slurm_script/slurm_eval_mpc_controllers.sh
#
# - Output and error logs will be written to: slurm_logs_eval/slurm-mpc-<jobid>.out/.err
# -------------------------------------------------------------------------------
