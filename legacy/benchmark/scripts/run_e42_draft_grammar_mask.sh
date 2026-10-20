#!/usr/bin/env bash
# E42 -- DRAFT-SIDE GRAMMAR MASK: does constraining the DRAFTER to the request's
# JSON schema buy anything, on top of the target-side mask vLLM already has?
#
# Both arms run structured output (the same JSON schema) AND MTP. They differ in
# exactly one thing, an environment variable:
#
#   maskoff -- upstream behaviour. The drafter samples argmax over unconstrained
#              logits; illegal drafts are removed post hoc by the scheduler's
#              `validate_tokens`, which truncates the chain at the first illegal
#              token.
#   maskon  -- UNIEINFRA_DRAFT_GRAMMAR_MASK=1. The drafter's logits are masked
#              with the request's grammar at every draft position, so it can
#              only propose schema-legal tokens.
#
# REPORTED METRIC IS ACCEPTED *LENGTH*, NOT ACCEPTANCE *RATE*. E40 showed the
# rate is not comparable across grammar settings: prefix truncation shrinks the
# denominator (submitted drafts), so the rate rises even when nothing improves.
# Length and end-to-end throughput are the honest statistics.
#
# k is swept because that is where a win could still hide: at small k the
# truncated tail is short, so losing it costs little.
set -uo pipefail

CONTAINER=${CONTAINER:-unieinfra-vllm-build}
GPU=${GPU:-3}
# The box is shared with other agents AND a killed engine does not release its
# 80 GiB instantly. Both produced spurious "Free memory on device is less than
# desired GPU memory utilization" aborts on the first attempt at this run, so
# every server start now waits for the card to actually be free.
NEED_FREE_MIB=${NEED_FREE_MIB:-88000}
# GPU 0 is another user's and is never touched. The rest of the box is shared
# with other agents, so the run pins whichever of these frees up first.
GPU_CANDIDATES=${GPU_CANDIDATES:-"3 1 2"}
PORT=${PORT:-8342}
MODEL=${MODEL:-/workspace/models/Qwen/Qwen3.5-9B}
REPO=/workspace/roy/UnieInfra
VENV=/workspace/roy/venvs/unieinfra-vllm-venv/bin
RESULTS=$REPO/benchmark/results/unieinfra
SRVLOG=$REPO/benchmark/results/_srvlogs/e42
DATA=/workspace/roy/bench-data/structured_toolcall.jsonl
SCHEMA=/workspace/roy/bench-data/toolcall_schema.json

KS=${KS:-"3 6"}
CONCS=${CONCS:-"1 16"}
REPS=${REPS:-3}
OLEN=${OLEN:-128}
WARMUPS=${WARMUPS:-8}

mkdir -p "$SRVLOG" "$RESULTS"
dex() { docker exec "$CONTAINER" bash -lc "$1"; }

# NOTE: another agent runs its own `unieinfra serve` in this same container, so
# a bare `pkill -f 'unieinfra serve'` would kill their job. Only ever kill the
# PID listening on OUR port.
PIDFILE=$SRVLOG/server.pid

# Cleanup discipline on a box with several other agents, copied from E43 because
# it is the only thing that works here. Two halves:
#
#   * NEVER kill a process this script did not start. The server is started
#     under `setsid` so its pid is a process-GROUP leader; `kill -- -PID` takes
#     the server and its EngineCore children and nothing else.
#   * Make sure THEIR cleanup cannot kill OURS. run_e39c / run_e40 / run_e41 all
#     do a container-wide `pkill -f '[u]nieinfra serve'`, and the container
#     shares the host PID namespace, so that pattern reaches our process. It
#     killed three of this run's servers mid-startup before the launch was
#     changed to `python -m vllm.entrypoints.cli.main serve`, which matches
#     neither `unieinfra serve` nor `vllm serve`.
kill_server() {
  local pid
  pid=$(dex "cat $PIDFILE 2>/dev/null" | tr -d '\r\n')
  [ -n "$pid" ] || return 0
  dex "kill -- -$pid 2>/dev/null; true"
  for _ in $(seq 1 60); do
    dex "kill -0 -- -$pid 2>/dev/null" || break
    sleep 2
  done
  dex "kill -9 -- -$pid 2>/dev/null; rm -f $PIDFILE; true"
  sleep 5
}

gpu_free_mib() {   # gpu_free_mib <index>
  local used total
  used=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --id="$1" --query-gpu=memory.total --format=csv,noheader,nounits)
  echo $(( total - used ))
}

# Block until GPU $GPU has NEED_FREE_MIB free. The box is shared with several
# other agents, so this can genuinely take a while; better to wait than to run
# the two arms on different cards.
wait_for_gpu() {
  for i in $(seq 1 720); do
    [ "$(gpu_free_mib "$GPU")" -ge "$NEED_FREE_MIB" ] && return 0
    [ $(( i % 60 )) -eq 0 ] && echo "  [waiting for gpu $GPU, $(( i * 5 ))s]"
    sleep 5
  done
  echo "  [gpu $GPU never freed up]"; return 1
}

# Pick the card ONCE, at the start, and pin it for the whole experiment. Every
# arm must run on the same silicon or the A/B is meaningless. GPU 0 is another
# user's and is never a candidate.
pick_gpu() {
  [ -n "${GPU_PINNED:-}" ] && return 0
  for _ in $(seq 1 720); do
    for cand in $GPU_CANDIDATES; do
      if [ "$(gpu_free_mib "$cand")" -ge "$NEED_FREE_MIB" ]; then
        GPU=$cand; GPU_PINNED=1
        echo "=== pinned GPU $GPU for the whole run ==="
        return 0
      fi
    done
    sleep 5
  done
  echo "no GPU ever became free"; return 1
}

start_server_once() {   # start_server_once <arm> <k> <log>
  local arm=$1 k=$2 log=$3
  local mask=0
  [ "$arm" = "maskon" ] && mask=1
  : > "$log"
  docker exec -d "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1
    export UNIEINFRA_DRAFT_GRAMMAR_MASK=$mask
    export UNIEINFRA_DRAFT_GRAMMAR_STATS=1 UNIEINFRA_DRAFT_GRAMMAR_STATS_EVERY=1000
    setsid bash -c 'echo \$\$ > $PIDFILE; exec $VENV/python3 -m vllm.entrypoints.cli.main serve $MODEL \
      --port $PORT --max-model-len 4096 --gpu-memory-utilization 0.85 --served-model-name q9 \
      --speculative-config \"{\\\"method\\\":\\\"mtp\\\",\\\"num_speculative_tokens\\\":$k}\"' \
      >> $log 2>&1"
  for _ in $(seq 1 300); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  [ready]"; return 0; }
    # Only fatal signatures. The engine logs harmless Tracebacks at startup
    # (e.g. an optional unieinfra_kernel autotune import), so matching bare
    # "Traceback" aborts a perfectly good server.
    # "KeyboardInterrupt: terminated" means somebody else's harness ran a bare
    # `pkill -f 'unieinfra serve'` and took our process down with it. That
    # happens on this shared box; treat it as retryable rather than hanging
    # here for the full startup timeout.
    grep -qaE "Engine core initialization failed|CUDA out of memory|EngineCore failed to start|KeyboardInterrupt: terminated" \
        "$log" 2>/dev/null && {
      echo "  [FAIL]"; grep -am1 "ValueError\|Error:\|KeyboardInterrupt" "$log"; return 1; }
    sleep 4
  done
  echo "  [TIMEOUT]"; return 1
}

start_server() {   # start_server <arm> <k> <log>; retries after a full drain
  for attempt in 1 2 3 4; do
    wait_for_gpu || return 1
    start_server_once "$@" && return 0
    echo "  [attempt $attempt failed; draining and retrying]"
    kill_server
  done
  return 1
}

EXTRA_JSON=$(python3 -c "
import json;print(json.dumps({'structured_outputs':{'json':json.load(open('$SCHEMA'))}}))")

run_one() {   # run_one <arm> <k> <rep> <conc>
  local arm=$1 k=$2 rep=$3 c=$4
  local label="e42_${arm}_k${k}_r${rep}"
  local outdir="$RESULTS/$label"
  local np=$(( c * 16 )); [ "$np" -lt 64 ] && np=64; [ "$np" -gt 256 ] && np=256
  mkdir -p "$outdir"
  dex "cd $REPO && $VENV/unieinfra bench serve \
      --backend openai --host 127.0.0.1 --port $PORT --model q9 --tokenizer $MODEL \
      --dataset-name custom --dataset-path $DATA --custom-output-len $OLEN \
      --num-prompts $np --max-concurrency $c --num-warmups $WARMUPS \
      --temperature 0 --seed 0 --disable-tqdm --save-detailed \
      --save-result --result-dir $outdir --result-filename c${c}.json --label $label \
      --extra-body '$EXTRA_JSON'" \
      > "$SRVLOG/bench_${label}_c${c}.log" 2>&1
  echo "  -> arm=$arm k=$k rep=$rep C=$c"
  grep -E "Output token throughput|Mean TPOT|Mean TTFT|Accept" \
      "$SRVLOG/bench_${label}_c${c}.log" | sed 's/^/       /'
}

echo "=== E42 draft grammar mask -- model $MODEL ==="
pick_gpu || exit 1
for k in $KS; do
  for rep in $(seq 1 "$REPS"); do
    if [ $(( rep % 2 )) -eq 1 ]; then ARMS="maskoff maskon"; else ARMS="maskon maskoff"; fi
    for arm in $ARMS; do
      log="$SRVLOG/server_${arm}_k${k}_r${rep}.log"
      kill_server
      echo "[k=$k rep=$rep arm=$arm] starting server"
      start_server "$arm" "$k" "$log" || { echo "SKIP $arm k=$k rep=$rep"; continue; }
      for c in $CONCS; do run_one "$arm" "$k" "$rep" "$c"; done
      # Dump the schema-truncation counters this server accumulated.
      dex "curl -s http://127.0.0.1:$PORT/v1/models >/dev/null" || true
      kill_server
      grep -h "unieinfra draft-grammar" "$log" | tail -2 || true
    done
  done
done
kill_server
echo "=== e42 done ==="
