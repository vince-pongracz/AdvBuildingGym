#!/usr/bin/env bash
# Launch TensorBoard with suppressed TF noise.
# Logs output to tensorboard_fireup.log, prints the URL, CTRL+C stops it.

# Usage: ./start_tensorboard.sh <logdir>
# Example: ./start_tensorboard.sh /hkfs/home/haicore/iai/dj0397/AdvBuildingGym/models/env_test1_small/ray/sac/

set -euo pipefail

echo -e
echo "Starting TensorBoard with logdir: $1"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <logdir>" >&2
    exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

LOGFILE="tensorboard_fireup.log"
echo "Logging TensorBoard output to: $LOGFILE"

tensorboard --logdir "$1" > "$LOGFILE" 2>&1 &
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
