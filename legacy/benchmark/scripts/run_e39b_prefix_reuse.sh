#!/usr/bin/env bash
# E39b — PREFIX REUSE: the first time this project measures the prefix cache at all.
#
# Every benchmark E1-E38 ran RandomDataset with the cache flushed between levels.
# RandomDataset prompts are independent random token ids, so they share NO prefix:
# the flush was almost beside the point, the *dataset* already guaranteed ~0 reuse.
# This script measures what reuse is worth when it actually exists.
#
# Model is Qwen3-8B, NOT Qwen3.5-9B: prefix caching is force-disabled on the
# 9B hybrid-GDN model (see E39 -- `enable_prefix_caching=False` on both arms
# there), so it cannot answer this question at all.
#
# Three arms:
#   cache_off  -- --no-enable-prefix-caching. The floor: no reuse is possible.
#   cold       -- cache ON, but /reset_prefix_cache immediately before the run,
#                 which is what E1-E38's "FLUSH between levels" was reaching for.
#   warm       -- cache ON, run repeated back-to-back with NO reset, so it can
#                 ride the previous run's cached prefixes.
#
# Two datasets:
#   prefix_repetition -- purpose-built shared-prefix workload; genuine reuse.
#   mtbench           -- what the owner asked for. NOTE, and this is a real
#                        limitation reported rather than papered over: this
#                        harness's MTBenchDataset takes ONLY turns[0] ("We create
#                        a single turn dataset for MT-Bench"), so it is NOT
#                        multi-turn and the only prefix it shares between requests
#                        is the chat template preamble.
#
# /reset_prefix_cache lives on the dev router -> VLLM_SERVER_DEV_MODE=1 required.
set -uo pipefail

CONTAINER=${CONTAINER:-unieinfra-vllm-build}
GPU=${GPU:-2}
PORT=${PORT:-8312}
MODEL=${MODEL:-/workspace/models/Qwen/Qwen3-8B}
REPO=/workspace/roy/UnieInfra
VENV=/workspace/roy/venvs/unieinfra-vllm-venv/bin
RESULTS=$REPO/benchmark/results/unieinfra
SRVLOG=$REPO/benchmark/results/_srvlogs/e39b

CONC=${CONC:-16}
REPS=${REPS:-3}
NUM_PROMPTS=${NUM_PROMPTS:-128}
OLEN=${OLEN:-256}
WARMUPS=${WARMUPS:-8}

mkdir -p "$SRVLOG"
dex() { docker exec "$CONTAINER" bash -lc "$1"; }

kill_server() {
  dex "pkill -f '[u]nieinfra serve' >/dev/null 2>&1; true"
  for _ in $(seq 1 60); do dex "pgrep -f '[u]nieinfra serve' >/dev/null 2>&1" || break; sleep 2; done
  sleep 5
}

start_server() {   # start_server <cachemode: on|off>
  # NOTE: keep these on separate `local` lines. Bash expands every word of a
  # command before running it, so `local mode=$1 log="...${mode}..."` would
  # expand ${mode} BEFORE mode is assigned -- unbound under `set -u`.
  local mode=$1
  local flag=""
  local log="$SRVLOG/server_${mode}.log"
  [ "$mode" = "off" ] && flag="--no-enable-prefix-caching"
  : > "$log"
  docker exec -d "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1 VLLM_SERVER_DEV_MODE=1
    exec $VENV/unieinfra serve $MODEL --port $PORT \
      --max-model-len 4096 --gpu-memory-utilization 0.85 --served-model-name q8 \
      $flag > $log 2>&1"
  for _ in $(seq 1 300); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  [ready] cache=$mode"; return 0; }
    grep -qE "Traceback \(most recent|CUDA out of memory|Killed" "$log" 2>/dev/null && {
      echo "  [FAIL] cache=$mode"; tail -20 "$log"; return 1; }
    sleep 4
  done
  echo "  [TIMEOUT] cache=$mode"; return 1
}

# prefix cache counters, as "queries hits"
counters() {
  dex "curl -s http://127.0.0.1:$PORT/metrics | awk '
    /^vllm:prefix_cache_queries_total/ {q=\$2}
    /^vllm:prefix_cache_hits_total/    {h=\$2}
    END {printf \"%d %d\", q+0, h+0}'"
}

# run_one <arm> <dataset> <rep>
run_one() {
  local arm=$1 ds=$2 rep=$3
  local label="e39b_${arm}_${ds}_r${rep}"
  local outdir="$RESULTS/$label" dsargs np
  case "$ds" in
    prefixrep) np=$NUM_PROMPTS
               dsargs="--dataset-name prefix_repetition --prefix-repetition-prefix-len 1024 \
                       --prefix-repetition-suffix-len 128 --prefix-repetition-num-prefixes 8 \
                       --prefix-repetition-output-len $OLEN" ;;
    # mt-bench has exactly 80 rows. Asking for more would OVERSAMPLE (duplicate)
    # prompts, and duplicates are perfect prefix-cache hits -- that would
    # manufacture the very reuse this experiment is trying to measure.
    mtbench)   np=80
               dsargs="--dataset-name hf --dataset-path philschmid/mt-bench --hf-split train \
                       --hf-output-len $OLEN --no-oversample" ;;
    *) echo "unknown dataset $ds"; return 1 ;;
  esac
  mkdir -p "$outdir"

  # `cold` resets the cache first; `warm` deliberately does not.
  if [ "$arm" = "cold" ]; then
    dex "curl -s -X POST http://127.0.0.1:$PORT/reset_prefix_cache >/dev/null" || true
  fi

  local before after q0 h0 q1 h1
  before=$(counters); q0=${before% *}; h0=${before#* }

  dex "cd $REPO && HF_HUB_DISABLE_PROGRESS_BARS=1 $VENV/unieinfra bench serve \
      --backend openai --host 127.0.0.1 --port $PORT --model q8 --tokenizer $MODEL \
      $dsargs --num-prompts $np --max-concurrency $CONC --num-warmups $WARMUPS \
      --temperature 0 --seed 0 --disable-tqdm \
      --save-result --result-dir $outdir --result-filename c${CONC}.json --label $label" \
      > "$SRVLOG/bench_${label}.log" 2>&1

  after=$(counters); q1=${after% *}; h1=${after#* }
  local dq=$(( q1 - q0 )) dh=$(( h1 - h0 ))
  local rate="n/a"
  [ "$dq" -gt 0 ] && rate=$(awk -v h="$dh" -v q="$dq" 'BEGIN{printf "%.2f", 100*h/q}')
  echo "{\"arm\":\"$arm\",\"dataset\":\"$ds\",\"rep\":$rep,\"queries\":$dq,\"hits\":$dh,\"hit_rate_pct\":\"$rate\"}" \
      > "$outdir/prefix_cache.json"
  echo "  -> arm=$arm ds=$ds rep=$rep  prefix-cache hit rate=${rate}%  (queries=$dq hits=$dh)"
  grep -E "Output token throughput|Mean TTFT" "$SRVLOG/bench_${label}.log" | sed 's/^/       /'
}

echo "=== E39b prefix reuse — GPU $GPU, model $MODEL, C=$CONC ==="

# Server mode is the OUTER loop deliberately: cache on/off is a launch flag, so
# rotating it per repeat would cost 6 model loads (~4 min each) to buy nothing --
# the arms cannot drift relative to each other within one server anyway.
echo "--- cache ON: cold (reset first) vs warm (no reset) ---"
kill_server; start_server on || exit 1
for rep in $(seq 1 "$REPS"); do
  for ds in prefixrep mtbench; do
    run_one cold "$ds" "$rep"
    run_one warm "$ds" "$rep"     # immediately repeated, cache NOT reset
  done
done

echo "--- cache OFF (the floor: no reuse possible) ---"
kill_server; start_server off || exit 1
for rep in $(seq 1 "$REPS"); do
  for ds in prefixrep mtbench; do run_one cache_off "$ds" "$rep"; done
done
kill_server

for d in "$RESULTS"/e39b_*/; do
  [ -d "$d" ] || continue
  python3 "$REPO/benchmark/summarize.py" "$d" "$d/summary.csv" unieinfra >/dev/null 2>&1
done
echo "=== e39b done ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
