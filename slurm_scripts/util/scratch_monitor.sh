#!/usr/bin/env bash
# Shared scratch-disk monitor for SLURM wrappers.
#
# Sourced by slurm_train_ray.sh / slurm_eval_ray.sh / slurm_train_ma.sh /
# slurm_train_sb.sh AFTER util/snapshot_mode.sh so RUN_DIR is available in
# snapshot mode.
#
# Spawns a single backgrounded sampling loop that, every
# SCRATCH_MONITOR_INTERVAL seconds (default 300), appends:
#   * UTC timestamp
#   * du -sh of this job's per-job scratch tmpdir
#   * du -sh of every Ray session dir found in that tmpdir + its logs/ and
#     object_spilling/ subdirs (this is "what your training writes")
#   * df -h of the node-wide /scratch (this is "global / shared")
# to scratch_usage.log placed:
#   * snapshot mode  : ${RUN_DIR}/scratch_usage.log   (alongside slurm_<jobid>.{out,err})
#   * legacy mode    : ${SLURM_SUBMIT_DIR}/scratch_usage-${SLURM_JOB_ID}.log
#
# Opt-out by exporting SCRATCH_MONITOR_DISABLE=1 before sbatch.
# Override interval with SCRATCH_MONITOR_INTERVAL=<seconds>.
#
# The background loop is disowned from the shell job table so a later `wait`
# in the caller does not block on it, and an EXIT trap kills it on script
# termination. This is a sourced script — it intentionally mutates the
# caller's environment. Do not execute it directly.

if [ "${SCRATCH_MONITOR_DISABLE:-0}" = "1" ]; then
  echo "[scratch_monitor] disabled via SCRATCH_MONITOR_DISABLE=1"
  return 0 2>/dev/null || exit 0
fi

SCRATCH_MONITOR_INTERVAL="${SCRATCH_MONITOR_INTERVAL:-300}"

# Log path: prefer the snapshot run dir (so it sits alongside the
# slurm_<jobid>.{out,err} files written by tools/snapshot/submit_snapshot.py),
# fall back to the submit dir with a job-id suffix to avoid clobbering across
# runs.
if [ -n "${RUN_DIR:-}" ]; then
  _scratch_monitor_log="${RUN_DIR}/scratch_usage.log"
else
  _scratch_monitor_log="${SLURM_SUBMIT_DIR:-$PWD}/scratch_usage-${SLURM_JOB_ID:-$$}.log"
fi
mkdir -p "$(dirname "${_scratch_monitor_log}")" 2>/dev/null || true

# Per-job scratch tmpdir. SLURM_TMPDIR is set when the cluster allocates
# tmpdir; otherwise fall back to the conventional path on this site.
_scratch_monitor_root="${SLURM_TMPDIR:-/scratch/slurm_tmpdir/job_${SLURM_JOB_ID:-unknown}}"

(
  while true; do
    {
      printf '\n--- %s ---\n' "$(date -u +%FT%TZ)"
      if [ -d "${_scratch_monitor_root}" ]; then
        printf 'job_tmpdir_total  : '
        du -sh "${_scratch_monitor_root}" 2>/dev/null | awk '{print $1}'
        for d in "${_scratch_monitor_root}"/ray/session_*/; do
          [ -d "$d" ] || continue
          printf 'ray_session       : '
          du -sh "$d" 2>/dev/null | awk '{print $1}'
          for sub in logs object_spilling runtime_resources; do
            if [ -d "${d}${sub}" ]; then
              printf '  %-15s : ' "${sub}"
              du -sh "${d}${sub}" 2>/dev/null | awk '{print $1}'
            fi
          done
        done
      else
        printf 'job_tmpdir_total  : <missing %s>\n' "${_scratch_monitor_root}"
      fi
      printf 'node_scratch (df) : '
      df -h /scratch 2>/dev/null \
        | awk 'NR==2 {printf "used=%s avail=%s use%%=%s mount=%s\n", $3, $4, $5, $6}'
    } >> "${_scratch_monitor_log}" 2>&1
    sleep "${SCRATCH_MONITOR_INTERVAL}"
  done
) &
SCRATCH_MONITOR_PID=$!
# Disown so the caller's `wait` (used to drain the awk stderr filter in
# slurm_train_ray.sh) does not block on our infinite loop.
disown "${SCRATCH_MONITOR_PID}" 2>/dev/null || true
trap 'kill "${SCRATCH_MONITOR_PID}" 2>/dev/null || true' EXIT

echo "=== Scratch monitor started ==="
echo "  log file          : ${_scratch_monitor_log}"
echo "  interval (s)      : ${SCRATCH_MONITOR_INTERVAL}"
echo "  scratch root      : ${_scratch_monitor_root}"
echo "  monitor pid       : ${SCRATCH_MONITOR_PID}"