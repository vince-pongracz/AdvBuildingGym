#!/usr/bin/env bash
# Author: Vince Pongracz
# Created: 2026-07-14 | Version: 1.0
# Description: Report GPFS quota usage (blocks + files) and a per-directory
#              disk-usage breakdown so the biggest space consumers are visible.

# -----------------------------------------------------------------------------
# Usage:
#   sbatch slurm_scripts/slurm_disk_usage.sh [DIR ...]
#
#   DIR ...   Directories to break down with du (default: $HOME).
#
# Examples:
#   sbatch slurm_scripts/slurm_disk_usage.sh
#   sbatch slurm_scripts/slurm_disk_usage.sh "$HOME/AdvBuildingGym"
#
# Notes:
# - CPU only, single core; du over ~1M files on GPFS takes minutes, not hours.
# - Quota check alone is instant — for a quick look without a job, run:
#     /usr/lpp/mmfs/bin/mmlsquota -u $USER --block-size auto hkfs-home
# - Home/work are GPFS (IBM Spectrum Scale), so mmlsquota is the right tool
#   (not lfs quota, which is Lustre-only).
#   Link: https://www.nhr.kit.edu/userdocs/horeka/filesystems/
# -----------------------------------------------------------------------------

# Link to SLURM params: https://www.nhr.kit.edu/userdocs/haicore/batch/

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/util/slurm-disk-usage-%j.out
#SBATCH --error=slurm_logs/util/slurm-disk-usage-%j.err
#SBATCH --job-name=disk-usage

set -euo pipefail

MMLSQUOTA=/usr/lpp/mmfs/bin/mmlsquota

echo "=== SLURM Resource Info ==="
echo "Node : $(hostname)"
echo "User : ${USER}"
echo "Date : $(date -Is)"

# Per-user quota on both GPFS devices. The relevant limit is the USR quota
# inside the group fileset (e.g. fileset 'iai' on hkfs-home).
for device in hkfs-home hkfs-work; do
  echo ""
  echo "=== GPFS user quota: ${device} ==="
  "${MMLSQUOTA}" -u "${USER}" --block-size auto "${device}" || echo "[WARN] mmlsquota failed for ${device}"
done

if command -v ws_list >/dev/null 2>&1; then
  echo ""
  echo "=== Workspaces (ws_list) ==="
  ws_list || echo "(no workspaces)"
fi

# Directory breakdown: block usage and inode counts, largest first.
# Both matter — the GPFS quota limits blocks AND number of files.
TARGETS=("${@:-${HOME}}")
for target in "${TARGETS[@]}"; do
  echo ""
  echo "=== Total size of ${target} ==="
  du -xsh "${target}" 2>/dev/null || true

  echo ""
  echo "=== Subdirectories of ${target} by size ==="
  { du -xh --max-depth=5 "${target}" 2>/dev/null || true; } | sort -rh

  echo ""
  echo "=== Subdirectories of ${target} by file count ==="
  { du -x --inodes --max-depth=5 "${target}" 2>/dev/null || true; } | sort -rn
done

echo ""
echo "Disk usage report completed successfully."

# -------------------------------------------------------------------------------
# Notes:
# - Make the script executable:
#     chmod +x slurm_scripts/slurm_disk_usage.sh
# - Submit from the repo root so the slurm_logs/util/ output path resolves:
#     sbatch slurm_scripts/slurm_disk_usage.sh
# - Output and error logs will be written to `slurm_logs/util/`.
# -------------------------------------------------------------------------------
