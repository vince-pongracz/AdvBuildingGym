#!/usr/bin/env bash
# Author: G. Demirel (demirelg), KIT
# Maintainer: gokhan.demirel@kit.edu
# Created: 2025-06-17 | Version: 1.0
# Description: Submit SLURM training jobs for all RL algorithm, reward, and obs combinations

# -----------------------------------------------------------------------------
# Submit SLURM training jobs for all combinations of algorithms, reward modes,
# and observation variants using slurm_train_rl_batch.sh
# -----------------------------------------------------------------------------

# Define configurations
ALGORITHMS=("ppo" "sac" "a2c" "ddpg")
#ALGORITHMS=("ppo" "sac" "a2c")
REWARD_MODES=("temperature" "combined") 
OBS_VARIANTS_TEMP=("T01" "T02" "T03" "T04")
OBS_VARIANTS_COMBINED=("C01" "C02" "C03" "C04")

# Base number of environments
NUM_ENVS_PARA=4
NUM_ENVS_SEQ=1

# Submit jobs
for ALG in "${ALGORITHMS[@]}"; do
  for MODE in "${REWARD_MODES[@]}"; do
    if [ "$MODE" == "temperature" ]; then
      OBS_VARIANTS=("${OBS_VARIANTS_TEMP[@]}")
    elif [ "$MODE" == "economic" ]; then
      OBS_VARIANTS=("${OBS_VARIANTS_ECO[@]}")
    elif [ "$MODE" == "combined" ]; then
      OBS_VARIANTS=("${OBS_VARIANTS_COMBINED[@]}")
    else
      echo "Unknown reward mode: $MODE"
      continue
    fi

    for OBS in "${OBS_VARIANTS[@]}"; do
      if [ "$ALG" == "td3" ] || [ "$ALG" == "ddpg" ]; then
        NUM_ENVS=$NUM_ENVS_SEQ
      else
        NUM_ENVS=$NUM_ENVS_PARA
      fi
      echo "Submitting: $ALG | $MODE | $OBS | envs=$NUM_ENVS"
      sbatch "$(dirname "$0")/slurm_train_rl_single.sh" "$ALG" "$NUM_ENVS" "$MODE" "$OBS"
      sleep 0.5
    done
  done
done

# -------------------------------------------------------------------------------
# Notes:
# - Make sure the script is executable:
#     chmod +x slurm_script/slurm_train_rl_batch.sh
#
# - To launch all training jobs:
#     ./slurm_script/slurm_train_rl_batch.sh
#
# - This script submits one job per combination of:
#     - RL algorithm: ppo, sac, a2c, ddpg
#     - Reward mode: temperature, combined
#     - Observation variant: T01–T04 (for temperature), C01–C04 (for combined)
#
# - The number of environments is automatically set:
#     - Parallel (4 envs): ppo, sac, a2c
#     - Sequential (1 env): ddpg, td3
#
# - Each job calls: slurm_train_rl_batch.sh <algorithm> <num_envs> <reward_mode> <obs_variant>
# -------------------------------------------------------------------------------
