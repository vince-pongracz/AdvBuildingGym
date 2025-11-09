#!/usr/bin/env bash
# Author: G. Demirel (demirelg), KIT
# Maintainer: gokhan.demirel@kit.edu
# Created: 2025-06-17 | Version: 1.0
# Description: SLURM evaluation script for single RL algorithm (PPO, SAC, DDPG, A2C)

#SBATCH --job-name=eval-%j
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs_eval/slurm-%x-%j.out
#SBATCH --error=slurm_logs_eval/slurm-%x-%j.err

set -euo pipefail

# --- Input argument (algorithm) ---
ALG="${1:-ppo}"  # default: ppo

# --- activate venv ---
PYTHON_ENV="../llec_env"
source "${PYTHON_ENV}/bin/activate"

# --- helper ---
fmt() { printf '%02d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }

OUTDOOR="data/LLEC_outdoor_temperature_5min_data.csv"
REWARD_MODES=("combined" "temperature")
OBS_COMBINED=(C01 C02 C03 C04)
OBS_TEMPERATURE=(T01 T02 T03 T04)

start_wall=$(date +%s); sum_secs=0

run_cmd() {
  local alg="$1"
  local reward="$2"
  local obs="$3"
  local args="$4"
  local t0=$(date +%s)
  echo "-> python run_evaluation.py --algorithms $alg --reward_mode $reward --obs_variant $obs $args"
  python run_evaluation.py --algorithms "$alg" --reward_mode "$reward" --obs_variant "$obs" $args
  local dt=$(( $(date +%s) - t0 ))
  sum_secs=$(( sum_secs + dt ))
  echo "   elapsed: $(fmt $dt)"
}

echo "Evaluating algorithm: $ALG"
for REWARD_MODE in "${REWARD_MODES[@]}"; do
  if [[ "$REWARD_MODE" == "combined" ]]; then
    OBS=("${OBS_COMBINED[@]}")
  else
    OBS=("${OBS_TEMPERATURE[@]}")
  fi

  for ob in "${OBS[@]}"; do
    echo ">>> $ALG | reward_mode=$REWARD_MODE | obs=$ob"
    run_cmd "$ALG" "$REWARD_MODE" "$ob" "--outdoor_temperature_path $OUTDOOR"
    run_cmd "$ALG" "$REWARD_MODE" "$ob" ""
    run_cmd "$ALG" "$REWARD_MODE" "$ob" "--prefer_best --outdoor_temperature_path $OUTDOOR"
    run_cmd "$ALG" "$REWARD_MODE" "$ob" "--prefer_best"
  done
done

wall_secs=$(( $(date +%s) - start_wall ))
echo; echo "Total wall-time : $(fmt $wall_secs)"
echo "Sum of run times: $(fmt $sum_secs)"
echo "Done."

# -------------------------------------------------------------------------------
# Notes:
# - Make sure this script is executable:
#     chmod +x slurm_script/slurm_eval_rl_single.sh
#
# - Submit the evaluation job with:
#     sbatch slurm_script/slurm_eval_rl_single.sh <algorithm>
#
# Arguments:
#   <algorithm>    One of: ppo, sac, ddpg, a2c (default: ppo)
#
# Functionality:
#   This script evaluates the given RL algorithm across:
#     - Reward modes: "combined", "temperature"
#     - Observation variants: C01–C04 (for combined), T01–T04 (for temperature)
#     - Evaluation flags: with/without --prefer_best and outdoor temperature input
#
# Example usage:
#     sbatch slurm_script/slurm_eval_rl_single.sh ppo
#     sbatch slurm_script/slurm_eval_rl_single.sh sac
#     sbatch slurm_script/slurm_eval_rl_single.sh ddpg
#     sbatch slurm_script/slurm_eval_rl_single.sh a2c
#
# Output:
#   Logs will be stored in:
#     slurm_eval/slurm-eval-<jobid>.out
#     slurm_eval/slurm-eval-<jobid>.err
# -------------------------------------------------------------------------------
