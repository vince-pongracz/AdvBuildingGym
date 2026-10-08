#!/usr/bin/env bash
# Submit eval jobs (10 episodes, --plot-all) for every snapshot whose dir name
# is dated on/after a cutoff (default 20260601). Each snapshot is re-run via
# tools.snapshot.submit_snapshot; outputs land under <snapshot>/runs/<kind>_<ts>/.
#
# Per snapshot, three INDEPENDENT eval kinds may be submitted:
#   --kind eval      when a Ray checkpoint exists (runs/train*/models/.../rllib_checkpoint.json).
#                    submit_snapshot auto-discovers the latest checkpoint.
#   --kind eval-sb   when an SB3 model exists (runs/train_sb*/models/**/*.zip)
#                    AND run_eval_sb.py is frozen in the snapshot (manifest).
#                    submit_snapshot auto-discovers the latest model .zip.
#   --kind eval-rbc  when rule-based control is "realised" in the snapshot's
#                    FROZEN code, i.e. the manifest lists BOTH the entry script
#                    run_eval_rule_based.py AND the adv_building_gym/rbc_strats/
#                    package. Older snapshots predate RBC and cannot run it.
# A snapshot may get several, one, or none of these (none => skipped).
#
# Usage:
#   ./submit_snapshot_evals.sh [CUTOFF] [-- <extra sbatch flags>]
#
#   CUTOFF        Inclusive lower bound on the snapshot timestamp prefix
#                 ("<YYYYMMDD>_<HHMMSS>"). Either a date (YYYYMMDD) or a
#                 concrete date and time (YYYYMMDD_HHMMSS); a truncated time
#                 is zero-padded, e.g. 20260601_23 -> 20260601_230000.
#                 Default: 20260601.
#
# Env overrides:
#   EPISODES=10           number of eval episodes (default 10)
#   SNAPSHOTS_DIR=...     snapshots root (default: <repo>/snapshots)
#   VENV=...              virtualenv to activate (default: /home/iai/dj0397/adv_env)
#   DRY_RUN=1             print the submit commands without running them
#   SBATCH_ARGS="..."     extra sbatch flags forwarded to each submission
#
# Examples:
#   ./submit_snapshot_evals.sh                       # all snapshots >= 20260601
#   ./submit_snapshot_evals.sh 20260610              # only the newest ones
#   ./submit_snapshot_evals.sh 20260525_1430         # from 2026-05-25 14:30:00 on
#   DRY_RUN=1 ./submit_snapshot_evals.sh             # preview, submit nothing
#   SBATCH_ARGS="--time=01:00:00" ./submit_snapshot_evals.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CUTOFF="${1:-20260601}"
EPISODES="${EPISODES:-10}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$REPO_ROOT/snapshots}"
VENV="${VENV:-/home/iai/dj0397/adv_env}"
SBATCH_ARGS="${SBATCH_ARGS:-}"

# Accept YYYYMMDD or YYYYMMDD_HHMMSS (time may be truncated, e.g. _23 or _1430).
# Missing time digits are zero-padded so the cutoff is inclusive from that instant,
# matching the fixed-width "<YYYYMMDD>_<HHMMSS>" snapshot-name prefix.
if ! [[ "$CUTOFF" =~ ^[0-9]{8}(_[0-9]{1,6})?$ ]]; then
    echo "ERROR: CUTOFF must be YYYYMMDD or YYYYMMDD_HHMMSS, got '$CUTOFF'" >&2
    exit 1
fi
cutoff_time="${CUTOFF:9}000000"
CUTOFF_TS="${CUTOFF:0:8}_${cutoff_time:0:6}"

# Activate the project virtualenv so `python -m tools.snapshot.submit_snapshot`
# resolves. Skip gracefully if already inside it.
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "WARNING: venv not found at $VENV; using current python" >&2
fi

echo "Cutoff      : >= $CUTOFF_TS"
echo "Episodes    : $EPISODES"
echo "Snapshots   : $SNAPSHOTS_DIR"
[[ -n "$SBATCH_ARGS" ]] && echo "sbatch args : $SBATCH_ARGS"
[[ "${DRY_RUN:-0}" == "1" ]] && echo "Mode        : DRY RUN (no submission)"
echo

# A Ray checkpoint exists to evaluate (mirrors submit_snapshot auto-discovery).
has_ray_checkpoint() {
    local snap="$1"
    find "$snap"/runs/train*/models -name rllib_checkpoint.json 2>/dev/null | grep -q .
}

# An SB3 model exists to evaluate (mirrors _find_latest_sb_train_model in
# tools/snapshot/submit_snapshot.py: any .zip under runs/train_sb*/models/).
has_sb_model() {
    local snap="$1"
    find "$snap"/runs/train_sb*/models -name '*.zip' 2>/dev/null | grep -q .
}

# SB eval runs the snapshot's FROZEN code, so run_eval_sb.py must be frozen in
# it. Probe the manifest's included_files without extracting snapshot.zip.
has_sb_eval_realised() {
    local snap="$1" mf="$snap/manifest.json"
    [[ -f "$mf" ]] || return 1
    grep -q '"path": "run_eval_sb.py"' "$mf"
}

# Rule-based eval is runnable only if the snapshot's FROZEN code contains both
# the entry script and the rbc_strats package. The manifest's included_files
# lists every frozen path, so probe it without extracting snapshot.zip.
has_rbc_realised() {
    local snap="$1" mf="$snap/manifest.json"
    [[ -f "$mf" ]] || return 1
    grep -q '"path": "run_eval_rule_based.py"' "$mf" \
        && grep -q '"path": "adv_building_gym/rbc_strats/' "$mf"
}

submit_eval() {
    local snap="$1" kind="$2"
    echo "  -> submit --kind $kind"
    local cmd=(python -m tools.snapshot.submit_snapshot --snapshot "$snap" --kind "$kind")
    [[ -n "$SBATCH_ARGS" ]] && cmd+=(--sbatch "$SBATCH_ARGS")
    cmd+=(-- --episodes "$EPISODES" --plot-all)
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '     %q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
}

n_eval=0
n_sb=0
n_rbc=0
n_skipped=0

for snap in "$SNAPSHOTS_DIR"/*/; do
    snap="${snap%/}"
    name="$(basename "$snap")"

    # Snapshot dirs are named "<YYYYMMDD>_<HHMMSS>_<trial>"; compare the full
    # timestamp prefix lexicographically (fixed-width, so string order = time order).
    ts_prefix="${name:0:15}"
    [[ "$ts_prefix" =~ ^[0-9]{8}_[0-9]{6}$ ]] || { echo "SKIP $name (no timestamp prefix)"; n_skipped=$((n_skipped + 1)); continue; }
    [[ "$ts_prefix" < "$CUTOFF_TS" ]] && continue

    did_submit=0
    echo "$name"
    if has_ray_checkpoint "$snap"; then
        submit_eval "$snap" eval
        n_eval=$((n_eval + 1)); did_submit=1
    fi
    if has_sb_model "$snap"; then
        if has_sb_eval_realised "$snap"; then
            submit_eval "$snap" eval-sb
            n_sb=$((n_sb + 1)); did_submit=1
        else
            echo "  -> SKIP eval-sb (SB model found but run_eval_sb.py not frozen in snapshot)"
        fi
    fi
    if has_rbc_realised "$snap"; then
        submit_eval "$snap" eval-rbc
        n_rbc=$((n_rbc + 1)); did_submit=1
    fi
    if [[ "$did_submit" -eq 0 ]]; then
        echo "  -> SKIP (no Ray checkpoint, no SB model, no RBC realised in snapshot)"
        n_skipped=$((n_skipped + 1))
    fi
done

echo
echo "Done. eval: $n_eval, eval-sb: $n_sb, eval-rbc: $n_rbc, skipped: $n_skipped"
