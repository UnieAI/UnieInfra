#!/usr/bin/env bash
# E40b — GENUINE MULTI-TURN PREFIX REUSE (conversation growth).
#
# E39b measured shared-prefix reuse (8 fixed prefixes x 1024 tokens, real and
# large: -64% TTFT cold, -83% warm) but could NOT measure conversation growth,
# because this tree's MTBenchDataset takes `turns[0]` and ConversationDataset
# takes `conv[0]`/`conv[1]` -- both are single-turn by construction.
#
# e40_multiturn_bench.py drives the real thing: within a session, turn k's
# prompt is [u1, a1, u2, a2, ..., uk], so the reusable prefix grows by a whole
# turn each round, and turns within a session are strictly sequential. Many
# sessions run concurrently.
#
# Prefix caching cannot be toggled at runtime, so the OFF and ON arms are
# separate server processes rather than interleaved runs. To make an order
# effect visible rather than assumed, the OFF arm is run again after the ON arm
# (rep 4, "confirm"); if the box drifted, that rep will show it.
#
#   off    server started --no-enable-prefix-caching
#   cold   caching on, /reset_prefix_cache immediately before the run
#   warm   caching on, the same conversations replayed with no reset
set -uo pipefail

CONTAINER=${CONTAINER:-unieinfra-vllm-build}
GPU=${GPU:-2}
PORT=${PORT:-8324}
MODEL=${MODEL:-/workspace/models/Qwen/Qwen3-8B}
REPO=/workspace/roy/UnieInfra
VENV=/workspace/roy/venvs/unieinfra-vllm-venv/bin
RESULTS=$REPO/benchmark/results/unieinfra/e40b
DATA=${DATA:-/workspace/roy/bench-data/mtbench_multiturn.jsonl}

SESSIONS=${SESSIONS:-32}
TURNS=${TURNS:-6}
CONC=${CONC:-16}
OLEN=${OLEN:-256}
REPS=${REPS:-3}

SRVLOG=$REPO/benchmark/results/_srvlogs/e40b
mkdir -p "$SRVLOG"
dex() { docker exec "$CONTAINER" bash -lc "$1"; }

kill_server() {
  dex "pkill -f '[u]nieinfra serve' >/dev/null 2>&1; true"
  for _ in $(seq 1 60); do dex "pgrep -f '[u]nieinfra serve' >/dev/null 2>&1" || break; sleep 2; done
  sleep 5
}

start_server() {   # start_server <cache_on 0|1> <logtag>
  local cache=$1 tag=$2
  local log="$SRVLOG/server_${tag}.log"; : > "$log"
  local pc="--no-enable-prefix-caching"
  [ "$cache" = "1" ] && pc=""
  docker exec -d "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1 VLLM_SERVER_DEV_MODE=1
    exec $VENV/unieinfra serve $MODEL --port $PORT \
      --max-model-len 8192 --gpu-memory-utilization 0.85 --served-model-name q8 $pc \
      --enable-prompt-tokens-details \
      > $log 2>&1"
  for _ in $(seq 1 300); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  [ready $tag]"; return 0; }
    grep -qE "Traceback \(most recent|CUDA out of memory|Killed" "$log" 2>/dev/null && {
      echo "  [FAIL]"; tail -30 "$log"; return 1; }
    sleep 4
  done
  echo "  [TIMEOUT]"; return 1
}

run_one() {   # run_one <arm> <rep> <reset 0|1>
  local arm=$1 rep=$2 reset=$3
  local label="e40b_${arm}_r${rep}"
  local extra=""
  [ "$reset" = "1" ] && extra="--reset-prefix-cache"
  dex "mkdir -p $RESULTS && cd $REPO && $VENV/python benchmark/scripts/e40_multiturn_bench.py \
      --host 127.0.0.1 --port $PORT --model q8 --data $DATA \
      --sessions $SESSIONS --turns $TURNS --max-concurrency $CONC --output-len $OLEN \
      --temperature 0 --seed 0 --arm $arm --label $label $extra \
      --out $RESULTS/${label}.json" 2>&1 | grep -vE "kernel selection layer" | sed 's/^/  /'
}

echo "=== E40b multi-turn — GPU $GPU sessions=$SESSIONS turns=$TURNS C=$CONC ==="

kill_server; start_server 0 off || exit 1
for rep in $(seq 1 "$REPS"); do echo "-- off rep $rep"; run_one off "$rep" 0; done

kill_server; start_server 1 on || exit 1
for rep in $(seq 1 "$REPS"); do
  echo "-- cache-on rep $rep"
  run_one cold "$rep" 1     # reset first: reuse can only come from THIS run
  run_one warm "$rep" 0     # replay: previous run's blocks are still resident
done

# drift check: the OFF arm again, after everything else
kill_server; start_server 0 off_confirm || exit 1
echo "-- off confirm rep 4"; run_one off 4 0
kill_server
echo "=== e40b done ==="
