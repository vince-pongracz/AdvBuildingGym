#!/usr/bin/env bash
# Description: Benchmark obs flattening — FlattenObservation wrapper (production,
#              env-side) vs FlattenObservations connector (retired, env-to-module),
#              timed per step over 3 full episodes. CPU only, 2 CPUs.
#
# Usage (submit from the repo root):
#   sbatch perf_test/slurm_bench_flatten.sh
#
# Logs land in perf_test/logs/ (kept with the results, unlike slurm_logs/ which is
# gitignored). The benchmark must run on a compute node with the repo on /hkfs/home —
# the login-node-local /scratch tree is NOT visible to compute nodes.
#
# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=perf_test/logs/slurm-bench-flatten-%j.out
#SBATCH --error=perf_test/logs/slurm-bench-flatten-%j.err
#SBATCH --job-name=bench-flatten

set -euo pipefail

cd /hkfs/home/haicore/iai/dj0397/AdvBuildingGym

# Activate virtual environment
PYTHON_ENV="../adv_env"
if [ -d "$PYTHON_ENV" ]; then
  source "${PYTHON_ENV}/bin/activate"
else
  echo "[WARN] Python environment not found at ${PYTHON_ENV}; continuing without activation"
fi

echo "=== SLURM Resource Info ==="
echo "Job ID              : ${SLURM_JOB_ID:-}"
echo "Node                : $(hostname)"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-}"
python --version

# Pin BLAS/OpenMP threads to the allocation so timings are reproducible and stay
# within the 2 requested CPUs (the flatten kernels are tiny, but numpy/torch would
# otherwise size their pools from the node's core count, not the cgroup).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1

echo "======"
echo "Running: python -u perf_test/bench_flatten_episodes.py"
python -u perf_test/bench_flatten_episodes.py

echo "Benchmark completed successfully."
