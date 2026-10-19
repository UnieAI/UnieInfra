#!/bin/bash
# AgentX cells on an already-running DeepSeek-V4.1 server. The prefix cache is
# flushed before every cell, so each cell starts with an empty cache; warmup uses
# a corpus that shares no prefix. Cells are equal-work (MIN_TOKENS) and greedy.
#   run_cells.sh BASE MODEL TAG FLUSH_PATH [CONCS]
set -u
BASE=$1; MODEL=$2; TAG=$3; FLUSH=$4; CONCS=${5:-"8 16 32"}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
H=$ROOT/bench/agent-slo
CORPORA=${CORPORA:-$H/corpora}
PY=${PYTHON:-python3}
OUT=${RESULTS:-$ROOT/unieinfra/bench/agentx/results}; mkdir -p "$OUT"
DATA=${DATA:-$CORPORA/swe_smith_agent_sess32x8.jsonl}
WARM=${WARM:-$CORPORA/short_decode_64.jsonl}
NREQ=${NREQ:-256}
for c in $CONCS; do
  code=$(curl -s -X POST "$BASE$FLUSH" -o /dev/null -w "%{http_code}")
  echo "flush $code"
  # A cell after an unflushed one measures a warm prefix cache, not the engine.
  [ "$code" = 200 ] || { echo "prefix cache flush failed ($code); aborting $TAG"; exit 1; }
  sleep 5
  [ -n "${PRE_CELL_HOOK:-}" ] && { bash "$PRE_CELL_HOOK" > "$OUT/${TAG}_c${c}.hook.txt" 2>&1 & }
  echo "== $TAG c=$c $(date +%T)"
  MIN_TOKENS=${MIN_TOKENS:-128} $PY "$H/drive.py" "$BASE" "$MODEL" "$DATA" "$c" "$NREQ" "$OUT/${TAG}_c${c}.json" "$WARM" 2>&1 | tail -3
  wait
  curl -s "$BASE/metrics" > "$OUT/${TAG}_c${c}.metrics.txt" 2>/dev/null
done
