#!/usr/bin/env bash
# Author: G. Demirel (demirelg), KIT
# Maintainer: gokhan.demirel@kit.edu
# Created: 2025-06-17 | Version: 1.0
# Description: Train RL agents (PPO, SAC, TD3, A2C) in the LLECBuildingGym environment

# -----------------------------------------------------------------------------
# Train RL agents (PPO, SAC, TD3, A2C) using the LLECBuildingGym environment.
#
# Usage:
#   sbatch slurm_train_rl_single.sh <ALGORITHM> <NUM_ENVS> <REWARD_MODE> <OBS_VARIANT>
#
# Example:
#   sbatch slurm_train_rl_single.sh ppo 4 combined C03
#
# Required Arguments:
#   <ALGORITHM>     - One of: ppo, sac, td3, a2c
#   <NUM_ENVS>      - Number of parallel environments (e.g. 4)
#   <REWARD_MODE>   - Reward mode (temperature | combined)
#   <OBS_VARIANT>   - Observation variant (e.g. T01–T04 or C01–C04)
# Additional Arguments:
# Only 2 parameters significantly accelerate RL training:
#
# #SBATCH --cpus-per-task=152
#   More CPUs => more parallel VecEnv workers (NUM_ENVS) => faster sampling => significantly faster RL training.
# #SBATCH --gres=gpu:4g.20gb:1
#   SB3 uses only one GPU => requesting more GPUs in the SB3 library does NOT improve training speed.
# -----------------------------------------------------------------------------

#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs_train/slurm-train-%j.out
#SBATCH --error=slurm_logs_train/slurm-train-%j.err
#SBATCH --job-name=rl-train-%j
#SBATCH --cpus-per-task=38

set -euo pipefail

# Activate Python virtual environment

# ABSOLUTE PATH:
# Select the path where the environment was installed.
# This method always works, especially inside SLURM jobs.
# Example:
#   source /home/xx1234/llec_env/bin/activate

# RELATIVE PATH:
# Works if you followed the README.md setup and your terminal or SLURM script runs:
#   cd LLECBuildingGym
PYTHON_ENV="../llec_env"
source "${PYTHON_ENV}/bin/activate"

# Parse input arguments
ALGORITHM=${1:?Missing algorithm (ppo|sac|td3|a2c)}
NUM_ENVS=${2:?Missing number of parallel environments}
REWARD_MODE=${3:?Missing reward mode (temperature|combined)}
OBS_VARIANT=${4:?Missing observation variant (T01–T04 or C01–C04)}
SEED=42
TIMESTEPS=1000000  # 1M timesteps

# Log configuration
echo "Starting training:"
echo "  Algorithm       : $ALGORITHM"
echo "  Environments    : $NUM_ENVS"
echo "  Reward mode     : $REWARD_MODE"
echo "  Observation     : $OBS_VARIANT"
echo "  Seed            : $SEED"
echo "  Timesteps       : $TIMESTEPS"

# Log SLURM resource allocation
echo "=== SLURM Resource Info ==="
echo "SLURM_CPUS_PER_TASK : $SLURM_CPUS_PER_TASK"
echo "CPU affinity        : $(taskset -pc $$ | awk -F: '{print $2}')"

# Log system and GPU info (for reproducibility in publications)
echo ""
echo "=== System Info ==="
echo "Hostname              : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"

echo ""
echo "=== GPU Info (nvidia-smi) ==="
nvidia-smi

echo ""
echo "=== PyTorch CUDA Info ==="
python3.9 -c "
import torch
print(f'CUDA available        : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device name (ID 0)    : {torch.cuda.get_device_name(0)}')
    print(f'CUDA version (torch)  : {torch.version.cuda}')
    print(f'CUDNN version         : {torch.backends.cudnn.version()}')
"

# Run the training script
python3.9 run_train_rl.py \
    --algorithm "$ALGORITHM" \
    --num-envs "$NUM_ENVS" \
    --reward_mode "$REWARD_MODE" \
    --obs_variant "$OBS_VARIANT" \
    --training \
    --seed "$SEED" \
    --timesteps "$TIMESTEPS"

echo "Training completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make sure the script is executable:
#     chmod +x slurm_script/slurm_train_01a_rl_single.sh
#
# - Submit with:
#     sbatch slurm_script/slurm_train_01a_rl_single.sh <algorithm> <num_envs> <reward_mode> <obs_variant>
#
# - Required arguments:
#     <algorithm>     : One of ppo, sac, td3, a2c
#     <num_envs>      : Number of parallel environments (e.g., 4 or 1)
#     <reward_mode>   : Reward mode ("temperature" or "combined")
#     <obs_variant>   : Observation variant (e.g., T01–T04 or C01–C04)
#
# - Example submissions:
#     sbatch slurm_script/slurm_train_01a_rl_single.sh a2c 4 combined C03
#     sbatch slurm_script/slurm_train_01a_rl_single.sh ppo 4 temperature T03
#
# - Output and error logs:
#     slurm_logs_train/slurm-train-<jobid>.out
#     slurm_logs_train/slurm-train-<jobid>.err
# -------------------------------------------------------------------------------
