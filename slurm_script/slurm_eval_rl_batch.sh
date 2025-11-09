#!/usr/bin/env bash
# Author: G. Demirel (demirelg), KIT
# Maintainer: gokhan.demirel@kit.edu
# Created: 2025-06-17 | Version: 1.0
# Description: Submits SLURM jobs for RL evaluation (PPO, SAC, DDPG, A2C)

set -euo pipefail

# List of RL algorithms to evaluate
ALGORITHMS=(ppo sac ddpg a2c)

# Starting counter for job numbering
count=1

# Loop over algorithms and submit a job for each
for ALG in "${ALGORITHMS[@]}"; do
  JOB_NAME=$(printf "eval-%02d" "$count")
  echo "Submitting job: $JOB_NAME for algorithm: $ALG"
  sbatch --job-name="$JOB_NAME" slurm_script/slurm_eval_rl_single.sh "$ALG"
  ((count++))
done

echo "All evaluation jobs submitted."

# -------------------------------------------------------------------------------
# Notes:
# - Make sure the script is executable:
#     chmod +x slurm_script/slurm_eval_rl_batch.sh
#
# - To run the script:
#     ./slurm_script/slurm_eval_rl_batch.sh
#
# - Adjust the algorithm list (ALGORITHMS) or naming (eval-XX) as needed.
# -------------------------------------------------------------------------------
