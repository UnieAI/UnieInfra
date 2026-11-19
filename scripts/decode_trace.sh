#!/bin/bash
# Kernel-level profile of long-context decode: attaches rocprofv3 to the live
# unieinfra scheduler for a few seconds while a fixed-shape decode load runs.
#
# Usage: decode_trace.sh [CONC] [PREFIX_TOKENS] [TRACE_SECS]
#   requires a unieinfra server already serving on :8000 (scripts/serve_unieinfra.sh)
#
# Why: the measured decode step at 120-200 K context is ~152 ms for 16 sequences,
# but every static traffic estimate for the sparse path (top-512 index, 2048
# candidate blocks) is 1-2 orders of magnitude smaller. Only a kernel trace can
# say where the time goes.
set -eo pipefail
ROOT=/home/smc/workspace/roy
CONC=${1:-16}
PREFIX=${2:-150000}
SECS=${3:-10}
OUT_TOKENS=${4:-64}
CTR=ts-dev
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/results/trace/decode-$STAMP
mkdir -p "$OUT"

curl -sf http://127.0.0.1:8000/health >/dev/null || {
  echo "no server on :8000 — start one with scripts/serve_unieinfra.sh first" >&2
  exit 2
}

# The scheduler lives inside the container; skip defunct entries from earlier runs.
PID=$(docker exec -u root "$CTR" bash -c \
  "ps -eo pid,state,comm --no-headers | awk '\$2 != \"Z\" && \$3 ~ /tokenspeed/ {print \$1}'" |
  head -1)
[[ -n "$PID" ]] || { echo "no live tokenspeed process in $CTR" >&2; exit 3; }
echo "attaching rocprofv3 to $CTR pid $PID for ${SECS}s; output $OUT"

docker exec -d -u root "$CTR" bash -c \
  "rm -rf /tmp/dectrace*; cd /tmp && /opt/rocm/bin/rocprofv3 -p $PID \
     --attach-duration-msec $((SECS * 1000)) --attach-sync-output \
     --kernel-trace -f csv -o /tmp/dectrace > /tmp/dectrace.log 2>&1"

# Give the attach a moment, then put the load on the server for longer than the
# trace window so the traced steps are steady state.
sleep $((SECS + 2))
python3 "$ROOT/scripts/decode_trace.py" run "$CONC" "$PREFIX" "$OUT_TOKENS" | tee "$OUT/client.log"

# Wait for the attach to detach and write its files.
for _ in $(seq 1 120); do
  docker exec -u root "$CTR" bash -c 'ls /tmp/dectrace*kernel_trace.csv >/dev/null 2>&1' && break
  sleep 2
done
docker cp "$CTR:/tmp/dectrace.log" "$OUT/rocprofv3.log" 2>/dev/null || true
docker exec -u root "$CTR" bash -c 'ls /tmp/dectrace* 2>/dev/null' > "$OUT/files.txt" || true
for f in $(cat "$OUT/files.txt" 2>/dev/null); do
  docker cp "$CTR:$f" "$OUT/$(basename "$f")" 2>/dev/null || true
done

CSV=$(ls "$OUT"/*kernel_trace.csv 2>/dev/null | head -1 || true)
if [[ -z "$CSV" ]]; then
  echo "no kernel trace produced; see $OUT/rocprofv3.log" >&2
  exit 4
fi
python3 "$ROOT/scripts/decode_trace.py" top "$CSV" 25 | tee "$OUT/top_kernels.txt"
