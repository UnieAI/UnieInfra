#!/bin/bash
# Capture and summarize a decode-only GPU trace from a running server.
#
#   ./profile_decode.sh PORT MODEL CONC DATA OUTDIR [SECONDS]
#
# Example: ./profile_decode.sh 8015 <model> 16 corpora/short_decode_64.jsonl \
#              ab_real/agentx-sweep/trace-c16 8
#
# Why this exists: every performance claim needs ms/step, kernels/step,
# us/kernel and device-busy measured on the same span (unieinfer/xref/
# 72-PARITY-STRATEGY.md §1). The engine already renders all four through its own
# torch profiler -- POST /start_profile on the control port (main + 1), whose
# body selects the output directory and the activities, then POST /stop_profile
# and it writes a chrome trace. Doing the measurement this way instead of
# editing timers into the hot path means the numbers cannot drift from the code
# they describe, and nothing temporary has to be removed afterwards.
#
# The load is decode-only on purpose: prefill would put extend kernels into the
# same step and blur what decode costs.
set -euo pipefail

PORT=${1:?port of the serving server}
MODEL=${2:?model path as the server knows it}
CONC=${3:?concurrency}
DATA=${4:?prompt corpus (jsonl with a "prompt" field)}
OUT=${5:?output directory for the trace}
DURATION=${6:-8}
WARM=${WARM:-4}

HERE=$(cd "$(dirname "$0")" && pwd)
CTRL_PORT=$((PORT + 1))
BASE="http://127.0.0.1:${PORT}"
CTRL="http://127.0.0.1:${CTRL_PORT}"
mkdir -p "$OUT"

# The engine refuses to profile while a previous trace is still being written;
# a stale one from an earlier attempt should not block this run.
if ! curl -sf "${CTRL}/health" >/dev/null; then
    echo "no control server on ${CTRL} -- is the engine up on ${PORT}?" >&2
    exit 1
fi

pids=()
cleanup() {
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT

# Keep decode in flight for the whole window. decode_only.py runs one bounded
# request set and exits, so it is restarted back to back; the trace only needs
# the server busy, not a single continuous request.
load_until=$((SECONDS + WARM + DURATION + 4))
while [ $SECONDS -lt $load_until ]; do
    python3 "$HERE/decode_only.py" "$BASE" "$MODEL" "$DATA" "$CONC" \
        "$OUT/load-${SECONDS}.json" >/dev/null 2>&1 &
    pids+=($!)
    sleep 1
done &

echo "warming ${WARM}s at concurrency ${CONC} ..." >&2
sleep "$WARM"

echo "profiling ${DURATION}s ..." >&2
curl -sf -X POST "${CTRL}/start_profile" -H 'content-type: application/json' \
    -d "{\"output_dir\": \"${OUT}\", \"activities\": [\"CPU\", \"GPU\"], \"with_stack\": true}" \
    >/dev/null || {
    echo "start_profile failed -- the control server proxies /start_profile to the engine;" >&2
    echo "check the control-port log for the reason (it is not fatal to the server)." >&2
    exit 1
}
sleep "$DURATION"
curl -sf -X POST "${CTRL}/stop_profile" >/dev/null || {
    echo "stop_profile failed; if a trace file still appeared it is usable" >&2
}

trace=$(ls -t "$OUT"/*.trace.json.gz 2>/dev/null | head -1 || true)
if [ -z "$trace" ]; then
    echo "no *.trace.json.gz in ${OUT}: profiling did not produce a trace" >&2
    exit 1
fi
echo "trace: ${trace}" >&2
python3 "$HERE/trace_stats.py" "$trace" --kernel mhc --json "${OUT}/trace-stats.json"
