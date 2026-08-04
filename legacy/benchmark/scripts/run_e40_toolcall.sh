#!/usr/bin/env bash
# E40 — TOOL-CALL QUALITY on real BFCL `tools`/`tool_choice` traffic.
#
# Two questions, one server config per invocation:
#
#   (1) How often does Qwen3-8B emit a tool call that fails to parse, or that
#       parses but violates the tool's JSON schema?  Measured on the DEFAULT
#       configuration everyone actually runs: `--enable-auto-tool-choice
#       --tool-call-parser hermes`, tool_choice="auto", no tool marked strict.
#       In that configuration vLLM builds NO grammar at all --
#       `tool_parsers/structural_tag_registry.py:get_model_structural_tag`
#       returns None for tool_choice="auto" unless some tool declares
#       `strict: true` -- so tool calls are recovered by regex from raw text.
#
#   (2) What does forcing the structural tag on buy, and what does it cost?
#       The `strict` arm injects `strict: true` into every tool, which is the
#       documented lever (docs/features/tool_calling.md).  Verified live, not
#       assumed: with an enum the prompt actively argues against, the non-strict
#       arm emits the prompt's value (schema violation) and the strict arm is
#       forced onto the enum.  Reproduce with benchmark/scripts/e40_grammar_probe.py;
#       recorded output in results/unieinfra/e40_probe/grammar_probe.json.
#
# Arms differ by EXACTLY ONE THING: the `strict` flag on the tools. Same frozen
# 256-request BFCL set, same greedy temperature 0, same server process.
#
# SPEC=1 additionally starts the server with an EAGLE3 draft model, which turns
# this into the grammar x speculative-decoding 2x2 when combined with a SPEC=0
# run. Acceptance is scraped from /metrics around each measured run.
set -uo pipefail

CONTAINER=${CONTAINER:-unieinfra-vllm-build}
GPU=${GPU:-2}
PORT=${PORT:-8323}
MODEL=${MODEL:-/workspace/models/Qwen/Qwen3-8B}
REPO=/workspace/roy/UnieInfra
VENV=/workspace/roy/venvs/unieinfra-vllm-venv/bin
RESULTS=$REPO/benchmark/results/unieinfra
DATA=${DATA:-/workspace/roy/bench-data/bfcl_256.jsonl}

SPEC=${SPEC:-0}                     # 1 => EAGLE3 speculative decoding on
SPEC_MODEL=${SPEC_MODEL:-AngelSlim/Qwen3-8B_eagle3}
SPEC_K=${SPEC_K:-3}
TAG=${TAG:-nospec}                  # results/log prefix, e.g. nospec | spec
CONCS=${CONCS:-"1 16"}
REPS=${REPS:-3}
OLEN=${OLEN:-256}
WARMUPS=${WARMUPS:-8}
NP_C1=${NP_C1:-64}                  # C=1 is serial; 64 prompts is ~40s/arm
NP_CN=${NP_CN:-256}                 # full frozen set at concurrency

SRVLOG=$REPO/benchmark/results/_srvlogs/e40
mkdir -p "$SRVLOG" "$RESULTS"
dex() { docker exec "$CONTAINER" bash -lc "$1"; }

kill_server() {
  dex "pkill -f '[u]nieinfra serve' >/dev/null 2>&1; true"
  for _ in $(seq 1 60); do dex "pgrep -f '[u]nieinfra serve' >/dev/null 2>&1" || break; sleep 2; done
  sleep 5
}

start_server() {
  local log="$SRVLOG/server_${TAG}.log"; : > "$log"
  local spec_arg=""
  if [ "$SPEC" = "1" ]; then
    spec_arg="--speculative-config '{\"method\":\"eagle3\",\"model\":\"$SPEC_MODEL\",\"num_speculative_tokens\":$SPEC_K}'"
  fi
  docker exec -d "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1
    exec $VENV/unieinfra serve $MODEL --port $PORT \
      --max-model-len 4096 --gpu-memory-utilization 0.85 --served-model-name q8 \
      --no-enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser hermes $spec_arg \
      > $log 2>&1"
  for _ in $(seq 1 300); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  [ready]"; return 0; }
    grep -qE "Traceback \(most recent|CUDA out of memory|Killed" "$log" 2>/dev/null && {
      echo "  [FAIL]"; tail -30 "$log"; return 1; }
    sleep 4
  done
  echo "  [TIMEOUT]"; return 1
}

run_one() {   # run_one <arm> <rep> <conc>
  local arm=$1 rep=$2 c=$3
  local label="e40_${TAG}_${arm}_c${c}_r${rep}"
  local outdir="$RESULTS/e40_${TAG}"
  local extra=""
  [ "$arm" = "strict" ] && extra="--strict"
  local np=$NP_CN; [ "$c" = "1" ] && np=$NP_C1
  dex "mkdir -p $outdir && cd $REPO && $VENV/python benchmark/scripts/e40_toolcall_bench.py \
      --host 127.0.0.1 --port $PORT --model q8 --data $DATA \
      --num-prompts $np --max-concurrency $c --output-len $OLEN \
      --temperature 0 --seed 0 --warmups $WARMUPS \
      --arm $arm --label $label $extra \
      --out $outdir/${label}.json" 2>&1 | grep -vE "kernel selection layer" | sed 's/^/  /'
}

echo "=== E40 tool-call quality — TAG=$TAG SPEC=$SPEC GPU=$GPU model=$MODEL ==="
kill_server
start_server || exit 1
for rep in $(seq 1 "$REPS"); do
  # rotate so the constrained arm is not always second
  if [ $(( rep % 2 )) -eq 1 ]; then ARMS="auto strict"; else ARMS="strict auto"; fi
  echo "-- rep $rep: $ARMS"
  for arm in $ARMS; do
    for c in $CONCS; do run_one "$arm" "$rep" "$c"; done
  done
done
kill_server
echo "=== e40 ($TAG) done ==="
