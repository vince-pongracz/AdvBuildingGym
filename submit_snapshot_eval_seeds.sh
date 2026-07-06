#!/usr/bin/env bash
# Submit eval jobs (10 episodes, --plot-all) across every (snapshot, seed) pair
# declared in a YAML sweep config. Companion to submit_snapshot_evals.sh (which
# scans every snapshot under a cutoff date); this one targets an explicit list of
# snapshots and fans each out across an explicit list of --seed overrides, feeding
# plotting/snapshot_eval_violins_seeds.ipynb.
#
# Each (snapshot, seed) pair is submitted independently via
# tools.snapshot.submit_snapshot --seed; outputs land under
# <snapshot>/runs/<kind>_<ts>_seed<N>/.
#
# Usage:
#   ./submit_snapshot_eval_seeds.sh <CONFIG_YAML>
#
# CONFIG_YAML schema:
#   seeds: [1, 2, 3, 4, 5]
#   snapshots:
#     - snapshots/20260703_004926_sta_lin_battery_only_price_sac
#     - snapshots/20260702_170441_sta_lin_battery_only_price_sb
#   kind: eval        # optional, default "eval" (eval / eval-sb / eval-rbc)
#   episodes: 10      # optional, default 10
#
# Env overrides:
#   VENV=...              virtualenv to activate (default: /home/iai/dj0397/adv_env)
#   DRY_RUN=1             print the submit commands without running them
#   SBATCH_ARGS="..."     extra sbatch flags forwarded to each submission
#
# Examples:
#   ./submit_snapshot_eval_seeds.sh configs/eval_sweeps/seed_sweep.yaml
#   DRY_RUN=1 ./submit_snapshot_eval_seeds.sh configs/eval_sweeps/seed_sweep.yaml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
    echo "Usage: $0 <CONFIG_YAML>" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

CONFIG_YAML="$1"
if [[ ! -f "$CONFIG_YAML" ]]; then
    echo "ERROR: config file does not exist: $CONFIG_YAML" >&2
    exit 1
fi

VENV="${VENV:-/home/iai/dj0397/adv_env}"
SBATCH_ARGS="${SBATCH_ARGS:-}"

# Activate the project virtualenv so `python -m tools.snapshot.submit_snapshot`
# resolves (and pyyaml is available for the config reads below). Skip
# gracefully if already inside it.
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "WARNING: venv not found at $VENV; using current python" >&2
fi

# Reads a required non-empty list key from the YAML config, one value per line.
_yaml_get_list() {
    local key="$1"
    python3 - "$CONFIG_YAML" "$key" <<'PY'
import sys
import yaml

path, key = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
values = data.get(key)
if not isinstance(values, list) or not values:
    print(f"ERROR: '{key}' must be a non-empty list in {path}", file=sys.stderr)
    sys.exit(1)
for v in values:
    print(v)
PY
}

# Reads an optional scalar key from the YAML config, falling back to a default.
_yaml_get_scalar() {
    local key="$1" default="$2"
    python3 - "$CONFIG_YAML" "$key" "$default" <<'PY'
import sys
import yaml

path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
print(data.get(key, default))
PY
}

mapfile -t SEEDS < <(_yaml_get_list seeds)
mapfile -t SNAPSHOTS < <(_yaml_get_list snapshots)
KIND="$(_yaml_get_scalar kind eval)"
EPISODES="$(_yaml_get_scalar episodes 10)"

for seed in "${SEEDS[@]}"; do
    if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "ERROR: seed must be an integer, got '$seed'" >&2
        exit 1
    fi
done

for snapshot in "${SNAPSHOTS[@]}"; do
    if [[ ! -d "$snapshot" ]]; then
        echo "ERROR: snapshot directory does not exist: $snapshot" >&2
        exit 1
    fi
done

echo "Config      : $CONFIG_YAML"
echo "Snapshots   : ${SNAPSHOTS[*]}"
echo "Seeds       : ${SEEDS[*]}"
echo "Kind        : $KIND"
echo "Episodes    : $EPISODES"
[[ -n "$SBATCH_ARGS" ]] && echo "sbatch args : $SBATCH_ARGS"
[[ "${DRY_RUN:-0}" == "1" ]] && echo "Mode        : DRY RUN (no submission)"
echo

n_submitted=0
for snapshot in "${SNAPSHOTS[@]}"; do
    echo "$snapshot"
    for seed in "${SEEDS[@]}"; do
        echo "  -> seed $seed"
        cmd=(python -m tools.snapshot.submit_snapshot --snapshot "$snapshot" --kind "$KIND" --seed "$seed")
        [[ -n "$SBATCH_ARGS" ]] && cmd+=(--sbatch "$SBATCH_ARGS")
        cmd+=(-- --episodes "$EPISODES" --plot-all)
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            printf '     %q ' "${cmd[@]}"; echo
        else
            "${cmd[@]}"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo
echo "Done. Submitted $n_submitted (snapshot, seed) job(s) across ${#SNAPSHOTS[@]} snapshot(s) and ${#SEEDS[@]} seed(s)."
