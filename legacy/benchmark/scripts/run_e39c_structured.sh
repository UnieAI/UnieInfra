#!/usr/bin/env bash
# E39c — STRUCTURED OUTPUT COST: what does a JSON schema constraint cost, and
# what does it buy in output validity?
#
# Same 256 tool-call-shaped prompts on both arms; the prompt ITSELF asks for a
# single JSON object with a fixed set of keys, so "malformed rate" is a fair
# question to ask of the unconstrained arm too (it was told what to produce).
#
#   free   -- no constraint. The model is asked for JSON in the prompt only.
#   json   -- the same request plus structured_outputs.json = the schema, so the
#             grammar backend masks logits and the output CANNOT leave the schema.
#
# Reported: output throughput / TTFT / TPOT cost of the constraint, and the
# malformed-output rate of each arm, computed by parsing `generated_texts`
# (kept by --save-detailed) and validating against the same schema.
set -uo pipefail

CONTAINER=${CONTAINER:-unieinfra-vllm-build}
GPU=${GPU:-2}
PORT=${PORT:-8313}
MODEL=${MODEL:-/workspace/models/Qwen/Qwen3-8B}
REPO=/workspace/roy/UnieInfra
VENV=/workspace/roy/venvs/unieinfra-vllm-venv/bin
RESULTS=$REPO/benchmark/results/unieinfra
SRVLOG=$REPO/benchmark/results/_srvlogs/e39c
DATA=/workspace/roy/bench-data/structured_toolcall.jsonl
SCHEMA=/workspace/roy/bench-data/toolcall_schema.json

CONCS=${CONCS:-"1 16"}
REPS=${REPS:-3}
OLEN=${OLEN:-128}
WARMUPS=${WARMUPS:-8}

mkdir -p "$SRVLOG"
dex() { docker exec "$CONTAINER" bash -lc "$1"; }

kill_server() {
  dex "pkill -f '[u]nieinfra serve' >/dev/null 2>&1; true"
  for _ in $(seq 1 60); do dex "pgrep -f '[u]nieinfra serve' >/dev/null 2>&1" || break; sleep 2; done
  sleep 5
}

start_server() {
  local log="$SRVLOG/server.log"; : > "$log"
  docker exec -d "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1
    exec $VENV/unieinfra serve $MODEL --port $PORT \
      --max-model-len 4096 --gpu-memory-utilization 0.85 --served-model-name q8 \
      > $log 2>&1"
  for _ in $(seq 1 300); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  [ready]"; return 0; }
    grep -qE "Traceback \(most recent|CUDA out of memory|Killed" "$log" 2>/dev/null && {
      echo "  [FAIL]"; tail -20 "$log"; return 1; }
    sleep 4
  done
  echo "  [TIMEOUT]"; return 1
}

EXTRA_JSON=$(python3 -c "
import json;print(json.dumps({'structured_outputs':{'json':json.load(open('$SCHEMA'))}}))")

run_one() {   # run_one <arm> <rep> <conc>
  local arm=$1 rep=$2 c=$3
  # separate lines: bash expands the whole command before assigning (see e39b)
  local label="e39c_${arm}_r${rep}"
  local outdir="$RESULTS/$label"
  local extra=""
  [ "$arm" = "json" ] && extra="--extra-body '$EXTRA_JSON'"
  # scale the prompt count with concurrency: 256 serial requests at C=1 would be
  # ~6 min per arm per repeat and buy no extra precision
  local np=$(( c * 16 )); [ "$np" -lt 64 ] && np=64; [ "$np" -gt 256 ] && np=256
  mkdir -p "$outdir"
  dex "cd $REPO && $VENV/unieinfra bench serve \
      --backend openai --host 127.0.0.1 --port $PORT --model q8 --tokenizer $MODEL \
      --dataset-name custom --dataset-path $DATA --custom-output-len $OLEN \
      --num-prompts $np --max-concurrency $c --num-warmups $WARMUPS \
      --temperature 0 --seed 0 --disable-tqdm --save-detailed \
      --save-result --result-dir $outdir --result-filename c${c}.json --label $label $extra" \
      > "$SRVLOG/bench_${label}_c${c}.log" 2>&1
  echo "  -> arm=$arm rep=$rep C=$c"
  grep -E "Output token throughput|Mean TTFT|Mean TPOT" "$SRVLOG/bench_${label}_c${c}.log" | sed 's/^/       /'
}

echo "=== E39c structured output — GPU $GPU, model $MODEL ==="
kill_server; start_server || exit 1
for rep in $(seq 1 "$REPS"); do
  # rotate so the constraint arm is not always second
  if [ $(( rep % 2 )) -eq 1 ]; then ARMS="free json"; else ARMS="json free"; fi
  for arm in $ARMS; do
    for c in $CONCS; do run_one "$arm" "$rep" "$c"; done
  done
done
kill_server

python3 "$REPO/benchmark/e39c_validate.py" --schema "$SCHEMA"
echo "=== e39c done ==="
