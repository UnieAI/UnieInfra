#!/usr/bin/env bash
# E48 — does fp8 KV cache actually deliver the +30% E44 projected, and what does
# it cost in accuracy?
#
# E44 measured the C=128 decode step as memory-bound: 35.65 GB of DRAM traffic in
# 28.51 ms = 1.251 TB/s sustained, 55.1% of it KV reads, with attention alone at
# 46.9% of the step AT 91.9% of peak DRAM.  A kernel already at 92% of the pipe
# cannot be tuned faster; it can only be fed fewer bytes.  Halving KV bytes was
# projected at ~+30% aggregate throughput -- a PROJECTION from measured
# bandwidth-bound behaviour, never a measurement.  E48 turns it into a number.
#
# THE CONFOUND THIS SCRIPT EXISTS TO CONTROL
# ------------------------------------------
# On sm_120 the fp8 arm CANNOT use the same attention kernel as the bf16 arm.
# `flash_attn_supports_kv_cache_dtype()` (`fa_utils.py:294`) requires FA3 on
# capability family 9.x or FA4 on family 10.x; sm_120 is family 12.x, so
# FlashAttention -- the backend E44 profiled, and the engine's default here --
# rejects fp8 KV outright, and `fp8_e5m2` is refused on every architecture
# (`fa_utils.py:284`).  So `--kv-cache-dtype fp8` silently changes the KERNEL as
# well as the DTYPE, and a naive bf16-vs-fp8 A/B measures the sum of the two.
#
# The arms therefore include a bf16 run pinned to the same backend the fp8 arms
# use.  That splits the observed delta into a backend cost and a dtype win:
#
#   bf16fa    kv=auto      backend=auto (-> FLASH_ATTN)   production baseline
#   bf16tri   kv=auto      backend=TRITON_ATTN            isolates the backend change
#   e4m3tri   kv=fp8_e4m3  backend=TRITON_ATTN            isolates the dtype change
#   e5m2tri   kv=fp8_e5m2  backend=TRITON_ATTN            mantissa-vs-range variant
#   e4m3auto  kv=fp8_e4m3  backend=auto                   what a user actually gets
#
# `bf16fa -> e4m3auto` is the product answer.  `bf16tri -> e4m3tri` is the
# physics answer.  Reporting only one of them would be dishonest in a different
# direction each way.
#
# PHASES (run in order, one GPU held across all of them)
#   base   one server per arm: KV pool capacity, the attention backend actually
#          selected, a greedy coherence probe, and GSM8K.  Answers "does it even
#          run", "how much more cache", "which kernel", "what accuracy".
#   tput   the throughput sweep, N reps, arms rotated back-to-back (E35).
#   long   ILEN 8192 / 32768, where KV bytes dominate and the bandwidth argument
#          predicts the gain GROWS.
#
# PROTOCOL (E35 discipline)
#   * N>=3 repeats, arms rotated, median with min/max spread, a delta inside the
#     spread is a TIE.
#   * ONE PROCESS PER ARM -- CUDA graphs are captured at engine construction.
#   * VLLM_DISABLE_COMPILE_CACHE=1 on every arm: vLLM's compile cache keys on
#     `VllmConfig`, not on env vars, and kv-cache-dtype changes graph structure.
#   * The client runs INSIDE the container (bridge-networked, no published
#     ports; the host venv lacks cbor2/msgspec/openai).
#   * The GPU is claimed ONCE and HELD by `e48_gpu_hold.py` for the whole sweep.
#     Grabbing scraps between arms is what produced discarded arms in E43/E44.
#   * Cleanup is by explicit PID only.  Servers start under `setsid` so
#     `kill -- -PID` reaps our process group and nothing else.  This script
#     never pattern-matches a kill: a container-wide `pkill` killed another
#     agent's server mid-benchmark earlier in this session.
#
# Usage:  [GPU=3] [REPS=3] [PHASES="base tput long"] run_e48_fp8_kv.sh
set -uo pipefail

REPO=/workspace/roy/UnieInfra
CONTAINER=${CONTAINER:-unieinfra-vllm-build}
CONTAINER_IP=${CONTAINER_IP:-172.17.0.6}
GPU=${GPU:-auto}
# Never GPU 0: it belongs to another user.
GPU_CANDIDATES=${GPU_CANDIDATES:-"3 1 2"}
PORT=${PORT:-8448}
REPS=${REPS:-3}
PHASES=${PHASES:-"base tput long"}
MODEL=${MODEL:-/workspace/models/Qwen/Qwen3-8B}
UNIE_BIN=/workspace/roy/venvs/unieinfra-vllm-venv/bin

ARMS=${ARMS:-"bf16fa bf16tri e4m3tri e5m2tri e4m3auto"}
RESULTS=${RESULTS:-$REPO/benchmark/results/e48}
SRVLOG=$REPO/benchmark/results/_srvlogs/e48
CONCS=${CONCS:-"1 32 128 256"}
OLEN=${OLEN:-256}
ILEN=${ILEN:-1024}
WARMUPS=${WARMUPS:-8}
MEMUTIL=${MEMUTIL:-0.85}
MAXLEN=${MAXLEN:-4096}
ATTEMPTS=${ATTEMPTS:-3}
GSM8K_LIMIT=${GSM8K_LIMIT:-1319}
HOLD_GIB=${HOLD_GIB:-4}

mkdir -p "$RESULTS" "$SRVLOG"
dex() { docker exec "$CONTAINER" bash -lc "$1"; }

PIDFILE=$SRVLOG/server.pid
HOLDPIDFILE=$SRVLOG/hold.pid

# arm_kvdt / arm_backend -- the two knobs that define an arm.
arm_kvdt() {
  case $1 in
    bf16fa|bf16tri|bf16fi) echo "auto" ;;
    e4m3tri|e4m3auto) echo "fp8_e4m3" ;;
    e5m2tri) echo "fp8_e5m2" ;;
    *) echo "auto" ;;
  esac
}
arm_backend() {
  case $1 in
    bf16tri|e4m3tri|e5m2tri) echo "TRITON_ATTN" ;;
    # `bf16fi` exists because the base phase showed that on sm_120 the AUTO path
    # for fp8 does not land on Triton at all -- FlashAttention is rejected
    # ("FP8 KV cache requires FA3 on SM90 or FA4 on SM100") and selection falls
    # through to FLASHINFER.  Without a bf16-on-FlashInfer control the headline
    # `bf16fa -> e4m3auto` delta is the sum of a backend change and a dtype
    # change, and there is no way to say which part is the fp8 win.
    bf16fi) echo "FLASHINFER" ;;
    *) echo "" ;;
  esac
}

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

# The sweep is long enough (~5 h) that it is run one phase at a time, so the
# hold has to survive between invocations: KEEP_HOLD=1 leaves the card claimed
# for the next phase.  Releasing it between phases is how a card gets lost.
stop_hold() {
  local pid
  [ "${KEEP_HOLD:-0}" = 1 ] && { echo "  [KEEP_HOLD] leaving GPU $GPU claimed"; return 0; }
  pid=$(cat "$HOLDPIDFILE" 2>/dev/null | tr -d '\r\n')
  [ -n "$pid" ] || return 0
  dex "kill -- -$pid 2>/dev/null; true"
  sleep 2
  dex "kill -9 -- -$pid 2>/dev/null; true"
  rm -f "$HOLDPIDFILE"
}

# Claim a card and keep it.  "Free" is checked twice, five seconds apart,
# because GPUs on this box get claimed between the check and the launch.
pick_gpu() {
  local g used used2
  for i in $(seq 1 720); do
    for g in $GPU_CANDIDATES; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      [ "$used" -lt 2000 ] || continue
      sleep 5
      used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      [ "$used2" -lt 2000 ] && { GPU=$g; echo "  [claimed GPU $g after $((i*10))s]"; return 0; }
    done
    [ "$i" = 1 ] && echo "  [CONTENTION] no free GPU among {$GPU_CANDIDATES} — waiting up to 2 h"
    sleep 10
  done
  echo "  [CONTENTION UNRESOLVED] no GPU came free"; return 1
}

start_hold() {
  local pid
  pid=$(cat "$HOLDPIDFILE" 2>/dev/null | tr -d '\r\n')
  if [ -n "$pid" ] && dex "kill -0 -- -$pid 2>/dev/null"; then
    echo "  [hold already running, pid $pid]"; return 0
  fi
  docker exec -d "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU
    setsid bash -c 'echo \$\$ > $HOLDPIDFILE; exec $UNIE_BIN/python3 $REPO/benchmark/scripts/e48_gpu_hold.py $GPU $HOLD_GIB' \
      >> $SRVLOG/hold.log 2>&1"
  for _ in $(seq 1 60); do
    grep -q "holding" "$SRVLOG/hold.log" 2>/dev/null && { echo "  [holding GPU $GPU with ${HOLD_GIB} GiB]"; return 0; }
    sleep 2
  done
  echo "  [WARN] hold process did not confirm; continuing without it"; return 0
}

# start_server <arm> <log> <maxlen>
start_server() {
  local arm=$1 log=$2 maxlen=$3
  local kvdt backend extra=""
  kvdt=$(arm_kvdt "$arm")
  backend=$(arm_backend "$arm")
  # `--attention-backend` folds into `attention_config.backend` (arg_utils.py:2275).
  # The dotted `--attention-config.backend=` form is NOT accepted by this pin.
  [ -n "$backend" ] && extra="--attention-backend $backend"
  : > "$log"
  docker exec -d "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1
    export UNIEINFRA_ATTENTION_LOG_SELECTION=1
    export UNIEINFRA_ATTENTION_SELECTION_DUMP=$SRVLOG/sel_${arm}
    setsid bash -c 'echo \$\$ > $PIDFILE; exec $UNIE_BIN/python3 -m vllm.entrypoints.cli.main serve $MODEL \
      --host 0.0.0.0 --port $PORT --served-model-name bench-model \
      --kv-cache-dtype $kvdt $extra \
      --max-model-len $maxlen --gpu-memory-utilization $MEMUTIL' >> $log 2>&1"
  for _ in $(seq 1 450); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  [ready] $arm"; return 0; }
    grep -qE "Traceback \(most recent|CUDA out of memory|Engine core initialization failed|Killed|ValueError" "$log" 2>/dev/null && {
      echo "  [FAIL] $arm — see $log"; tail -30 "$log"; return 1; }
    sleep 4
  done
  echo "  [TIMEOUT] $arm"; return 1
}

# Capacity + which kernel actually ran.  This is deliverable (2) and (5): a
# "working" fp8 flag that dequantizes the cache on read would give the capacity
# win with NONE of the bandwidth win, and the two are indistinguishable from
# throughput alone.
record_config() {
  local arm=$1 log=$2 out=$RESULTS/config_${arm}.txt
  {
    echo "arm=$arm kv_cache_dtype=$(arm_kvdt "$arm") forced_backend=$(arm_backend "$arm")"
    grep -E "GPU KV cache size|Maximum concurrency|Available KV cache memory|KV cache dtype|Using .* backend|attention backend|unieinfra_attention" "$log" | head -30
  } > "$out"
  cat "$out"
}

greedy_probe() {
  local arm=$1
  dex "curl -s http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"bench-model\",\"prompt\":\"The capital of Italy is\",\"max_tokens\":48,\"temperature\":0}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"choices\"][0][\"text\"])'" \
    | tee "$RESULTS/probe_${arm}.txt"
}

# run_one <arm> <rep> <C> <ilen> <tag>
# NP_MULT is lowered for the long-context phases: at ILEN 32768 a prompt costs
# 32x the prefill of the main sweep, and the question there is steady-state
# throughput under KV pressure, not sample count.
run_one() {
  local arm=$1 rep=$2 c=$3 ilen=$4 tag=$5 np mult
  mult=${NP_MULT:-8}
  np=$(( c * mult )); [ "$np" -gt 512 ] && np=512; [ "$np" -lt 16 ] && np=16
  dex "
    cd $REPO
    $UNIE_BIN/vllm bench serve --backend openai \
      --host 127.0.0.1 --port $PORT --model bench-model --tokenizer $MODEL \
      --dataset-name random --random-input-len $ilen --random-output-len $OLEN \
      --ignore-eos --num-prompts $np --max-concurrency $c --num-warmups $WARMUPS \
      --temperature 0 --seed 0 --disable-tqdm \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99 \
      --save-result --result-dir $RESULTS/${tag}_${arm}_r${rep} \
      --result-filename c${c}.json --label ${tag}_${arm}_r${rep}
  " >> "$SRVLOG/bench_${tag}_${arm}_r${rep}_c${c}.log" 2>&1
}

run_gsm8k() {
  local arm=$1
  dex "
    cd $REPO
    $UNIE_BIN/python3 benchmark/scripts/e48_gsm8k.py \
      --host 127.0.0.1 --port $PORT --model bench-model \
      --limit $GSM8K_LIMIT --concurrency 32 --label $arm \
      --out $RESULTS/gsm8k_${arm}.json
  " 2>&1 | tail -5
}

trap 'echo "[trap] cleaning up"; kill_server; stop_hold' EXIT INT TERM

echo "=== E48 fp8 KV cache validation ($(date -u +%FT%TZ)) ==="
echo "  arms:   $ARMS"
echo "  phases: $PHASES"
[ "$GPU" = "auto" ] && { pick_gpu || exit 1; }
echo "  GPU=$GPU"
start_hold

# --- phase base: capacity, backend evidence, coherence, accuracy -------------
if [[ " $PHASES " == *" base "* ]]; then
  for arm in $ARMS; do
    echo "--- base / $arm $(date -u +%FT%TZ)"
    kill_server
    ok=0
    for attempt in $(seq 1 "$ATTEMPTS"); do
      start_server "$arm" "$SRVLOG/base_${arm}.log" "$MAXLEN" && { ok=1; break; }
      echo "  [retry $attempt/$ATTEMPTS] $arm"; kill_server
    done
    if [ "$ok" != 1 ]; then
      echo "  [UNSUPPORTED-OR-BROKEN] $arm — recorded as a result, not skipped"
      { echo "arm=$arm kv_cache_dtype=$(arm_kvdt "$arm") forced_backend=$(arm_backend "$arm")";
        echo "STARTUP FAILED"; grep -E "ValueError|NotImplementedError|Error" "$SRVLOG/base_${arm}.log" | head -10; } \
        > "$RESULTS/config_${arm}.txt"
      continue
    fi
    record_config "$arm" "$SRVLOG/base_${arm}.log"
    greedy_probe "$arm"
    run_gsm8k "$arm"
    kill_server
  done
fi

# --- phase tput: the headline sweep -----------------------------------------
if [[ " $PHASES " == *" tput "* ]]; then
  for rep in $(seq 1 "$REPS"); do
    order="$ARMS"
    # Rotate arm order between reps so drift in box conditions is spread across
    # arms rather than confounded with one of them.
    [ $((rep % 2)) -eq 0 ] && order=$(echo "$ARMS" | tr ' ' '\n' | tac | tr '\n' ' ')
    for arm in $order; do
      [ -s "$RESULTS/config_${arm}.txt" ] && grep -q "STARTUP FAILED" "$RESULTS/config_${arm}.txt" && continue
      echo "--- tput rep $rep / $arm $(date -u +%FT%TZ)"
      kill_server
      ok=0
      for attempt in $(seq 1 "$ATTEMPTS"); do
        start_server "$arm" "$SRVLOG/tput_${arm}_r${rep}.log" "$MAXLEN" && { ok=1; break; }
        echo "  [retry $attempt/$ATTEMPTS] $arm"; kill_server
      done
      [ "$ok" = 1 ] || { echo "  [GIVING UP] $arm rep $rep"; continue; }
      for c in $CONCS; do run_one "$arm" "$rep" "$c" "$ILEN" "tput"; done
      kill_server
    done
  done
fi

# --- phase long: where KV bytes dominate ------------------------------------
if [[ " $PHASES " == *" long "* ]]; then
  LONG_REPS=${LONG_REPS:-3}
  for rep in $(seq 1 "$LONG_REPS"); do
    order="$ARMS"
    [ $((rep % 2)) -eq 0 ] && order=$(echo "$ARMS" | tr ' ' '\n' | tac | tr '\n' ' ')
    for arm in $order; do
      [ -s "$RESULTS/config_${arm}.txt" ] && grep -q "STARTUP FAILED" "$RESULTS/config_${arm}.txt" && continue
      echo "--- long rep $rep / $arm $(date -u +%FT%TZ)"
      kill_server
      ok=0
      for attempt in $(seq 1 "$ATTEMPTS"); do
        start_server "$arm" "$SRVLOG/long_${arm}_r${rep}.log" "${LONG_MAXLEN:-40960}" && { ok=1; break; }
        echo "  [retry $attempt/$ATTEMPTS] $arm"; kill_server
      done
      [ "$ok" = 1 ] || { echo "  [GIVING UP] $arm rep $rep"; continue; }
      [ "$rep" = 1 ] && record_config "${arm}_long" "$SRVLOG/long_${arm}_r${rep}.log"
      # ILEN 8192: C=32 needs 32 x 8448 tok x 144 KiB = 38.9 GB of bf16 KV, which
      # still fits the 63.7 GiB bf16 pool, so this level compares like-for-like.
      for c in ${LONG8K_CONCS:-"8 32"}; do NP_MULT=${NP_MULT_8K:-4} run_one "$arm" "$rep" "$c" 8192 "long8k"; done
      # ILEN 32768: C=4 fits both pools (4 x 4.5 GB).  C=16 needs ~72 GB of bf16
      # KV -- MORE than the whole bf16 pool -- but only ~36 GB of fp8 KV.  That
      # level is where the capacity win and the bandwidth win stop being
      # separable from the user's point of view, and it is the point of the
      # long-context phase.
      for c in ${LONG32K_CONCS:-"4 16"}; do NP_MULT=${NP_MULT_32K:-2} run_one "$arm" "$rep" "$c" 32768 "long32k"; done
      kill_server
    done
  done
fi

kill_server
stop_hold
echo "=== E48 done $(date -u +%FT%TZ) ==="
python3 "$REPO/benchmark/scripts/e48_summarize.py" "$RESULTS" || true
