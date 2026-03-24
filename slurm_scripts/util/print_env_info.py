#!/usr/bin/env python3
"""Print Python and hardware environment info for SLURM job logs.

Called by slurm_train_ray.sh and slurm_eval_ray.sh to provide consistent
environment diagnostics without duplicating the inline heredoc in each script.
"""

import os
import sys

print('Python executable :', sys.executable)
print('Python version    :', sys.version.splitlines()[0])

print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>'))
print('CUDA_MODULE_LOADING :', os.environ.get('CUDA_MODULE_LOADING', '<not set>'))
ld_path = os.environ.get('LD_LIBRARY_PATH', '<not set>')
print('LD_LIBRARY_PATH     :', ld_path[:200] + '...' if len(ld_path) > 200 else ld_path)

import torch
print('PyTorch version     :', torch.__version__)
print('CUDA built with     :', torch.version.cuda)

try:
    if torch.cuda.is_available():
        print('CUDA available      : True')
        print('Device count        :', torch.cuda.device_count())
        torch.cuda.init()
        print('CUDA init           : SUCCESS')
        print('Device name         :', torch.cuda.get_device_name(0))
        print('CUDNN version       :', torch.backends.cudnn.version())
    else:
        print('CUDA available      : False')
except Exception as e:
    print(f'CUDA init FAILED    : {type(e).__name__}: {e}')
