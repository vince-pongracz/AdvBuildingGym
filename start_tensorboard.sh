#!/usr/bin/env bash
# Launch TensorBoard with suppressed TF noise.
# Logs output to tensorboard_fireup.log, prints the URL, CTRL+C stops it.

# Usage: ./start_tensorboard.sh <logdir> [extra tensorboard args...]
# Example usages:
# ./start_tensorboard.sh /hkfs/home/haicore/iai/dj0397/AdvBuildingGym/models/env_test1_small/ray/sac/
# ./start_tensorboard.sh /hkfs/home/haicore/iai/dj0397/AdvBuildingGym/ep_metrics/eval_trajectories
# ./start_tensorboard.sh <logdir> --port 6007 --bind_all

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <logdir> [extra tensorboard args...]" >&2
    exit 1
fi

LOGDIR="$1"
shift

echo ""
echo "Starting TensorBoard with logdir: $LOGDIR"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

LOGFILE="tensorboard_fireup.log"
echo "Logging TensorBoard output to: $LOGFILE"

# --load_fast=false avoids the experimental fast loader, which can stall the
# UI when event files are still being written to by a live training job.
tensorboard --logdir "$LOGDIR" "$@" > "$LOGFILE" 2>&1 &
TB_PID=$!
trap 'kill "$TB_PID" 2>/dev/null; exit 0' INT TERM

for _ in $(seq 1 120); do
    if URL=$(grep -oP 'http://\S+' "$LOGFILE" 2>/dev/null); then
        echo "TensorBoard running at: $URL (PID: $TB_PID) — CTRL+C to stop"
        wait "$TB_PID"
    fi
    sleep 1
done

echo "ERROR: TensorBoard did not start within 120s. Check $LOGFILE" >&2
kill "$TB_PID" 2>/dev/null
exit 1
