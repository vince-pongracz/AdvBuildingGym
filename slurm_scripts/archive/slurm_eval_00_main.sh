#!/usr/bin/env bash
# Author: G. Demirel (demirelg), KIT
# Maintainer: gokhan.demirel@kit.edu
# Created: 2025-06-17 | Version: 1.0
# Description: Submits SLURM jobs for controller evaluation (RBC, MPC, RL Batch)

set -euo pipefail

# List of evaluation scripts and corresponding job names (submitted via SBATCH)
EVAL_JOBS=(
  "slurm_eval_01_rbc.sh:eval-rbc"
  "slurm_eval_02_mpc.sh:eval-mpc"
)

echo "==================== Evaluation Main ===================="
echo "Submitting evaluation jobs (RBC, MPC, RL)..."
echo "----------------------------------------------------------"

# Submit RBC and MPC as separate SLURM jobs
for ENTRY in "${EVAL_JOBS[@]}"; do
  IFS=":" read -r SCRIPT JOB_NAME <<< "$ENTRY"
  echo "Submitting job: ${JOB_NAME} using script: slurm_script/${SCRIPT}"
  sbatch --job-name="${JOB_NAME}" "slurm_script/${SCRIPT}"
done

echo "Submitting RL batch evaluations via shell (this script will call sbatch for singles)..."
bash slurm_script/slurm_eval_03_rl_batch.sh

echo "All evaluation jobs submitted."

# -------------------------------------------------------------------------------
# Notes:
# - Place this script in the same directory as the other SLURM scripts, e.g.:
#     slurm_script/slurm_eval_00_main.sh
#
# - Make sure the script is executable:
#     chmod +x slurm_script/slurm_eval_00_main.sh
#
# - To run the script from the project root:
#     ./slurm_script/slurm_eval_00_main.sh
#
# - The RL batch script (slurm_eval_03_rl_batch.sh) can internally call
#   slurm_eval_03a_rl_single.sh for per-algorithm evaluation.
# -------------------------------------------------------------------------------