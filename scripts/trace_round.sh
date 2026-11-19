#!/bin/bash
# One off-queue window that answers "where does a long-context decode step go?".
# Starts a unieinfra server at the c64 cap16 recipe (the cell that loses to vLLM
# 0.39x), runs scripts/decode_trace.sh against it, then stops everything.
#
# Why this cannot wait: every remaining scheduling lever is closed, and the only
# measured fact about the decode step is its wall time. The static traffic
# estimate for the sparse path (top-512 index over 2048 candidate blocks of 8)
# is 1-2 orders of magnitude smaller than the observed step, so the gap has to
# be in the kernels -- a per-kernel trace is the only way to name it.
#
# Usage: trace_round.sh [CONC] [PREFIX_TOKENS] [TRACE_SECS]
#   default 16 / 150000 / 10  -> the c64 cap16 shape
set -uo pipefail
ROOT=/home/smc/workspace/roy
CONC=${1:-16}
PREFIX=${2:-150000}
SECS=${3:-10}
LOG=$ROOT/logs/trace-round.log
mkdir -p "$ROOT/logs" "$ROOT/results/trace"

# serve_unieinfra.sh reads CONC and CAP; cap = CONC/4 is the recipe for c64.
export MAX_NUM_SEQS=$(( CONC > 4 ? CONC / 4 : 1 ))
echo "$(date -u +%FT%TZ) trace_round: serving unieinfra CONC=$CONC cap=$MAX_NUM_SEQS" >> "$LOG"
"$ROOT/scripts/serve_unieinfra.sh" "$CONC" >> "$LOG" 2>&1 || {
  echo "$(date -u +%FT%TZ) trace_round: server failed to start" >> "$LOG"
  "$ROOT/scripts/stop_servers.sh" >> "$LOG" 2>&1
  exit 1
}

# Wait for /health, up to 900 s (a cold 40-layer V4.1 with 3 MTP stages is slow).
ok=0
for _ in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then ok=1; break; fi
  sleep 5
done
if (( ok == 0 )); then
  echo "$(date -u +%FT%TZ) trace_round: /health never came up" >> "$LOG"
  "$ROOT/scripts/stop_servers.sh" >> "$LOG" 2>&1
  exit 1
fi
echo "$(date -u +%FT%TZ) trace_round: server READY, tracing" >> "$LOG"

"$ROOT/scripts/decode_trace.sh" "$CONC" "$PREFIX" "$SECS" >> "$LOG" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) trace_round: decode_trace rc=$rc" >> "$LOG"
"$ROOT/scripts/stop_servers.sh" >> "$LOG" 2>&1
exit $rc
