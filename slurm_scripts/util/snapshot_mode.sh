#!/usr/bin/env bash
# Shared snapshot-mode setup for SLURM wrappers.
#
# Sourced by slurm_train_ray.sh / slurm_eval_ray.sh / slurm_train_ma.sh /
# slurm_train_sb.sh. When the wrapper is invoked through
# tools/snapshot/submit_snapshot.py the following env vars are set by sbatch
# --export=:
#
#   SNAPSHOT_DIR     Absolute path to the snapshot dir
#   SNAPSHOT_RUN_ID  Per-submission run id (e.g. train_20260521_232358)
#   LIVE_REPO_ROOT   Absolute path to the live repo (provides plotting/)
#
# Usage from a SLURM wrapper:
#
#   ENTRY_SCRIPT_BASENAME="run_train_ray.py"   # only used in snapshot mode
#   # shellcheck source=util/snapshot_mode.sh
#   source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/util/snapshot_mode.sh"
#
# After sourcing, ENTRY_SCRIPT is set to either:
#   * <SNAPSHOT_DIR>/code/<basename>  in snapshot mode (CWD also moved to
#     <SNAPSHOT_DIR>/runs/<SNAPSHOT_RUN_ID>/ and PYTHONPATH prepended with
#     LIVE_REPO_ROOT so non-snapshotted packages like plotting/ resolve)
#   * <basename>                      in legacy live-repo mode
#
# This is a sourced script — it intentionally mutates the caller's
# environment. Do not execute it directly.

if [ -z "${ENTRY_SCRIPT_BASENAME:-}" ]; then
  echo "[snapshot_mode] ENTRY_SCRIPT_BASENAME must be set before sourcing this file" >&2
  return 1 2>/dev/null || exit 1
fi

if [ -n "${SNAPSHOT_DIR:-}" ]; then
  # Serialise first-run extraction across concurrent jobs. Without the lock,
  # two jobs starting against a brand-new snapshot at the same moment can both
  # pass the existence check and unzip on top of each other. Acquiring an
  # uncontended lock is microseconds, so this stays cheap on subsequent runs.
  (
    flock -x 9
    if [ ! -d "${SNAPSHOT_DIR}/code" ]; then
      echo "=== Extracting snapshot.zip into ${SNAPSHOT_DIR}/code/ ==="
      unzip -q -d "${SNAPSHOT_DIR}/code" "${SNAPSHOT_DIR}/snapshot.zip"
    fi
  ) 9>"${SNAPSHOT_DIR}/.code.lock"

  # Symlink the live repo's data/ into the snapshot's code/ dir. Required
  # because adv_building_gym/devices/statesources/csv_loader.py resolves
  # relative CSV paths against Path(__file__).parents[3] (= the dir
  # containing the adv_building_gym package = <SNAPSHOT_DIR>/code/) so its
  # Ray-worker CSV reads can succeed regardless of CWD. Without this link
  # workers fail with: FileNotFoundError <SNAPSHOT_DIR>/code/data/...
  # Re-applied each run so a moved snapshot picks up a new LIVE_REPO_ROOT.
  if [ -n "${LIVE_REPO_ROOT:-}" ] && [ -d "${LIVE_REPO_ROOT}/data" ]; then
    ln -sfn "${LIVE_REPO_ROOT}/data" "${SNAPSHOT_DIR}/code/data"
  fi

  RUN_DIR="${SNAPSHOT_DIR}/runs/${SNAPSHOT_RUN_ID:?SNAPSHOT_RUN_ID must be set in snapshot mode}"
  mkdir -p "${RUN_DIR}"
  cd "${RUN_DIR}"

  # Wire CWD-relative path resolution. The trial config + data/-/infra-/
  # statesource-schedule YAMLs reference paths like `configs/...` and
  # `data/...` that get tested by Path(...).exists() at driver-side load
  # time, which uses CWD = RUN_DIR. Symlink them so:
  #   * configs/  -> snapshot's frozen configs   (immutable)
  #   * data/     -> live repo's CSV data        (intentionally shared)
  # Outputs (models/, ep_metrics/, eval_results/, result_*.json) are created
  # as real directories under RUN_DIR by the entry script.
  ln -sfn "${SNAPSHOT_DIR}/code/configs" configs
  if [ -n "${LIVE_REPO_ROOT:-}" ] && [ -d "${LIVE_REPO_ROOT}/data" ]; then
    ln -sfn "${LIVE_REPO_ROOT}/data" data
  else
    echo "[snapshot_mode] LIVE_REPO_ROOT/data not available; CSV paths will not resolve" >&2
  fi

  ENTRY_SCRIPT="${SNAPSHOT_DIR}/code/${ENTRY_SCRIPT_BASENAME}"
  # Live repo provides plotting/ (deliberately not snapshotted) — see
  # tools/snapshot/make_snapshot.py for the whitelist.
  export PYTHONPATH="${LIVE_REPO_ROOT:-}${PYTHONPATH:+:${PYTHONPATH}}"
  echo "=== Snapshot mode ==="
  echo "  SNAPSHOT_DIR        : ${SNAPSHOT_DIR}"
  echo "  RUN_DIR (CWD)       : ${RUN_DIR}"
  echo "  ENTRY_SCRIPT        : ${ENTRY_SCRIPT}"
  echo "  PYTHONPATH          : ${PYTHONPATH}"
  echo "  configs/         -> $(readlink configs)"
  echo "  data/            -> $(readlink data 2>/dev/null || echo '(not linked)')"
  echo "  code/data/       -> $(readlink "${SNAPSHOT_DIR}/code/data" 2>/dev/null || echo '(not linked)')"
else
  ENTRY_SCRIPT="${ENTRY_SCRIPT_BASENAME}"
fi
