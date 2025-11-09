#!/usr/bin/env bash
# Author: G. Demirel (demirelg), KIT
# Maintainer: gokhan.demirel@kit.edu
# Created: 2025-06-17 | Version: 1.0
# Description: GPU availability and configuration check using nvidia-smi and PyTorch

#SBATCH --job-name=gpu-info
#SBATCH --output=slurm_logs_train/gpu-info-%j.out
#SBATCH --error=slurm_logs_train/gpu-info-%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=00:05:00

set -euo pipefail

PYTHON_ENV="../llec_env"
source "${PYTHON_ENV}/bin/activate"

echo "=== GPU Hardware Info (nvidia-smi) ==="
nvidia-smi

echo ""
echo "=== PyTorch CUDA Info ==="
python3.9 -c "
import torch
print(f'CUDA available        : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count          : {torch.cuda.device_count()}')
    print(f'Device name (ID 0)    : {torch.cuda.get_device_name(0)}')
    print(f'CUDA version (torch)  : {torch.version.cuda}')
    print(f'CUDNN version         : {torch.backends.cudnn.version()}')
else:
    print('No CUDA-enabled GPU accessible.')
"

# -------------------------------------------------------------------------------
# Notes:
# - Make sure the script is executable:
#     chmod +x slurm_script/slurm_check_gpu_info.sh
#
# - To submit the job to SLURM:
#     sbatch slurm_script/slurm_check_gpu_info.sh
#
# - Output and error logs will be written to: slurm/gpu-info-<jobid>.out/.err
# -------------------------------------------------------------------------------
