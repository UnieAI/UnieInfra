#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the UnieInfra project
#
# E49 -- UnieInfra out-of-the-box vs vLLM out-of-the-box, plus the attribution
# arms that say which half of any win is ours.
#
# ARM DESIGN (why there are four, not two)
# ----------------------------------------
# The headline the owner wants is "UnieInfra's defaults vs vLLM's defaults".
# But two arms cannot distinguish three different causes of a delta: our
# defaults, our kernel seams, and the engine version.  So:
#
#   A  unieinfra   our engine, our defaults          <- the product
#   B  vllm        our engine, upstream defaults,    <- isolates DEFAULTS only
#                  UnieInfra kernel seams OFF           (same binary as A)
#   C  vllm        arm B + the same features turned  <- isolates "our defaults"
#                  on by hand                           from "features vLLM has"
#   D  stock       a separate stock vLLM install,    <- external reference
#                  out of the box
#
# A vs B is the clean product claim: identical binary, identical hardware, the
# only variable is which console-script name was typed.  A vs C answers "is the
# win ours, or did we just switch on what vLLM already had?".  D keeps us
# honest against a real upstream install, and is the only arm whose engine
# version differs -- so a D delta must never be attributed to defaults.
#
# BOX RULES (learned the hard way, see E43/E47)
#  * one GPU is claimed and HELD for the whole sweep, so no co-tenant can take
#    it between arms (one process per arm is mandatory: CUDA graphs are
#    captured at engine construction);
#  * every server is started under `setsid` and killed with `kill -- -PID` on
#    its own process group ONLY.  Never pattern-match: a container-wide
#    `pkill -f` killed another agent's server mid-run during E47.
#
# Usage:
#   run_e49_product_ab.sh <model-path> [gpu] [outdir]

set -euo pipefail

MODEL="${1:?usage: run_e49_product_ab.sh <model-path> [gpu] [outdir]}"
GPU="${2:-2}"
TAG="$(basename "$MODEL")"
OUTDIR="${3:-/workspace/roy/UnieInfra/benchmark/results/e49/$TAG}"

REPS="${REPS:-3}"
CONCURRENCIES="${CONCURRENCIES:-1 16 32 64 128 256}"
ILEN="${ILEN:-1024}"
OLEN="${OLEN:-256}"
MAXLEN="${MAXLEN:-4096}"
UTIL="${UTIL:-0.85}"
TP="${TP:-1}"
PORT="${PORT:-8491}"

UNIE_VENV=/workspace/roy/venvs/unieinfra-vllm-venv/bin
STOCK_VENV=/home/ubuntu/vllm-bench-venv/bin
SRVLOG="$OUTDIR/_srvlogs"
mkdir -p "$OUTDIR" "$SRVLOG"

# The features arm C turns on by hand.  These mirror exactly what the defaults
# layer chooses for a checkpoint that ships MTP weights -- except for the depth
# ladder, which stock 0.25.1 has no field for (see the note in the log entry).
KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-32}"
SPEC_K="${SPEC_K:-3}"

log() { printf '[e49 %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# --------------------------------------------------------------------------
# GPU claim-and-hold.  Released by the EXIT trap, by PID, never by pattern.
# --------------------------------------------------------------------------
HOLD_PID=""
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping server process group $SERVER_PID"
    kill -- "-$SERVER_PID" 2>/dev/null || true
    sleep 5
    kill -9 -- "-$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$HOLD_PID" ]] && kill -0 "$HOLD_PID" 2>/dev/null; then
    log "releasing GPU hold $HOLD_PID"
    kill "$HOLD_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

claim_gpu() {
  CUDA_VISIBLE_DEVICES="$GPU" setsid "$UNIE_VENV/python3" \
    /workspace/roy/UnieInfra/benchmark/scripts/e48_gpu_hold.py 0 4 \
    >"$SRVLOG/gpu_hold.log" 2>&1 &
  HOLD_PID=$!
  sleep 10
  log "GPU $GPU claimed and held by PID $HOLD_PID"
}

wait_for_server() {
  local port=$1 pid=$2 deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log "server PID $pid died during startup"
      return 1
    fi
    sleep 5
  done
  log "server did not become healthy within 30 min"
  return 1
}

# start_server <arm> <logfile>
start_server() {
  local arm=$1 logfile=$2
  local -a cmd env_prefix

  case "$arm" in
    A)  # our engine, our defaults -- the product
        cmd=("$UNIE_VENV/unieinfra" serve "$MODEL")
        env_prefix=()
        ;;
    B)  # our engine, upstream defaults. The kernel seams are left at their
        # product setting (on) so that A vs B differs by the DEFAULTS and
        # nothing else -- disabling them here would fold the E37/E43/E47 kernel
        # deltas into what is supposed to be a defaults measurement. Arm D is
        # where a seams-free engine is compared.
        #
        # ARM_B_EXTRA exists for one honest reason: on a hybrid model, upstream
        # defaults can FAIL TO BOOT. `max_num_seqs` defaults to 1024 while the
        # GDN/Mamba state pool only affords ~500 blocks, and
        # `resolve_cudagraph_mode_and_sizes()` raises rather than clamping.
        # Arm A dodges that check only because its adaptive spec ladder has
        # already forced cudagraph_mode to PIECEWISE -- an accident, not a
        # virtue. Passing `--max-num-seqs 500` is the minimum intervention that
        # lets the control arm start at all, and it is NON-BINDING at every
        # concurrency this sweep tests (max 128 << 500), so it cannot flatter
        # either arm. Recorded in the log entry rather than hidden.
        cmd=("$UNIE_VENV/vllm" serve "$MODEL")
        if [[ -n "${ARM_B_EXTRA:-}" ]]; then
          # shellcheck disable=SC2206
          cmd+=($ARM_B_EXTRA)
        fi
        env_prefix=()
        ;;
    C)  # arm B plus the equivalent features, enabled by hand.
        # NOTE: stock vLLM 0.25.1 has no `num_speculative_tokens_per_batch_size`
        # field, so this arm can only pin a static depth. The adaptive ladder is
        # therefore NOT comparable between A and C, and the log entry says so.
        cmd=("$UNIE_VENV/vllm" serve "$MODEL")
        env_prefix=()
        if [[ "${ARM_C_OFFLOAD:-auto}" != "off" ]]; then
          cmd+=(--kv-offloading-size "$KV_OFFLOAD_GIB")
        fi
        if [[ "${ARM_C_MTP:-auto}" != "off" ]]; then
          cmd+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K}")
        fi
        ;;
    D)  # a genuinely separate stock vLLM install, out of the box
        cmd=("$STOCK_VENV/vllm" serve "$MODEL")
        env_prefix=()
        ;;
    *)  log "unknown arm $arm"; return 1 ;;
  esac

  cmd+=(--host 127.0.0.1 --port "$PORT" --served-model-name bench
        --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL"
        --tensor-parallel-size "$TP")

  # VLLM_DISABLE_COMPILE_CACHE=1: the compile/AOT cache keys on VllmConfig, so
  # arms that differ in graph structure would otherwise replay the first graph.
  # VLLM_SERVER_DEV_MODE=1 exposes /reset_prefix_cache so each rep starts cold.
  setsid env CUDA_VISIBLE_DEVICES="$GPU" VLLM_DISABLE_COMPILE_CACHE=1 \
    VLLM_SERVER_DEV_MODE=1 "${env_prefix[@]}" \
    "${cmd[@]}" >"$logfile" 2>&1 </dev/null &
  SERVER_PID=$!
  log "arm $arm server PID $SERVER_PID -> $logfile"
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 24); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 5
    done
    kill -9 -- "-$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  sleep 15
}

# Scrape spec-decode counters so we can PROVE MTP engaged rather than
# silently no-op'ing.  A default that quietly does nothing is worse than one
# that is off.
scrape_spec_metrics() {
  local outfile=$1
  curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null | awk '
    /^vllm:spec_decode_num_draft_tokens_total/     {d=$2}
    /^vllm:spec_decode_num_accepted_tokens_total/  {a=$2}
    /^vllm:prefix_cache_queries_total/             {q=$2}
    /^vllm:prefix_cache_hits_total/                {h=$2}
    END {printf "{\"draft_tokens\":%d,\"accepted_tokens\":%d,\"prefix_queries\":%d,\"prefix_hits\":%d}\n",
         d+0, a+0, q+0, h+0}' > "$outfile"
}

run_arm() {
  local arm=$1 rep=$2
  local logfile="$SRVLOG/${arm}_r${rep}.log"

  start_server "$arm" "$logfile"
  if ! wait_for_server "$PORT" "$SERVER_PID"; then
    log "arm $arm rep $rep FAILED to start; see $logfile"
    stop_server
    return 1
  fi

  local venv="$UNIE_VENV"
  [[ "$arm" == "D" ]] && venv="$STOCK_VENV"

  for c in $CONCURRENCIES; do
    local np=$(( c * 8 ))
    (( np < 64 )) && np=64

    # A hung cell must cost minutes, not an hour. E49 lost 52 minutes to a
    # client blocked on a request the server would never answer (the server had
    # been told to shut down mid-run and deadlocked draining one in-flight
    # request), with the GPU at 0% the whole time and nothing in any log to say
    # so. Re-check liveness before each cell and cap the client.
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      log "arm $arm rep $rep: server not healthy before C=$c -- aborting arm"
      stop_server
      return 1
    fi

    log "arm $arm rep $rep C=$c (n=$np)"
    curl -s -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1 || true
    scrape_spec_metrics "$OUTDIR/${arm}_r${rep}_c${c}_metrics_before.json"

    timeout --signal=INT --kill-after=60 "${CELL_TIMEOUT:-1500}" \
    "$venv/vllm" bench serve \
      --backend openai --host 127.0.0.1 --port "$PORT" \
      --model bench --tokenizer "$MODEL" \
      --dataset-name random --random-input-len "$ILEN" \
      --random-output-len "$OLEN" --ignore-eos \
      --num-prompts "$np" --max-concurrency "$c" \
      --num-warmups 8 --temperature 0 --seed 0 --disable-tqdm \
      --save-result --result-dir "$OUTDIR" \
      --result-filename "${arm}_r${rep}_c${c}.json" \
      --label "${arm}_r${rep}_c${c}" \
      >>"$SRVLOG/${arm}_r${rep}_client.log" 2>&1 || \
      log "arm $arm rep $rep C=$c client FAILED (continuing)"

    scrape_spec_metrics "$OUTDIR/${arm}_r${rep}_c${c}_metrics_after.json"
  done

  stop_server
}

# --------------------------------------------------------------------------
main() {
  log "model=$MODEL gpu=$GPU tp=$TP reps=$REPS out=$OUTDIR"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader > "$OUTDIR/_gpu_before.txt"
  claim_gpu

  local arms="${ARMS:-A B C D}"
  for rep in $(seq 1 "$REPS"); do
    # Rotate arm order every rep so a monotonic drift in box conditions cannot
    # be mistaken for an arm effect.
    local ordered="$arms"
    if (( rep % 2 == 0 )); then
      ordered=$(echo "$arms" | tr ' ' '\n' | tac | tr '\n' ' ')
    fi
    for arm in $ordered; do
      run_arm "$arm" "$rep" || true
    done
  done

  log "sweep complete -> $OUTDIR"
  log "summarize with: python benchmark/scripts/e49_summarize.py $OUTDIR"
}

main "$@"
