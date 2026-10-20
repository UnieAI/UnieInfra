#!/usr/bin/env bash
# E51 — stock vLLM defaults vs UnieInfra defaults (MTP + KV-fp8) on RTX PRO 6000.
#
# Four arms so every number is attributable, not just a headline:
#   stock      upstream `vllm serve`, nothing added
#   mtp        stock + MTP speculative decoding      (isolates the latency lever)
#   fp8kv      stock + fp8 KV cache                  (isolates the capacity/bandwidth lever)
#   unieinfra  `unieinfra serve` — what a user actually gets from us
#
# Arms run STRICTLY SEQUENTIALLY on one GPU. Three prior benchmark rounds in this
# project were corrupted by concurrent arms sharing a port, a result filename, or a
# health endpoint, so isolation here is deliberate and not negotiable: one server at
# a time, its PID tracked explicitly, its port probed for identity before any client
# starts, and a result file whose name carries the arm.
set -uo pipefail

GPU="${GPU:-1}"
MODEL="${MODEL:-/workspace/models/Qwen/Qwen3.6-35B-A3B}"
PORT_BASE="${PORT_BASE:-8420}"
PORT=0
OUT="${OUT:-/workspace/roy/UnieInfra/benchmark/results/e51}"
ILEN="${ILEN:-1024}"
OLEN="${OLEN:-256}"
CONCURRENCIES="${CONCURRENCIES:-1 8 32 128}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"

mkdir -p "$OUT"
SERVER_PID=""

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }

stop_server() {
    [ -z "$SERVER_PID" ] && return 0
    # Kill by the exact PID we started. Never pattern-match: this box hosts other
    # users' engines and a broad match would take them down.
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 90); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 -- "-$SERVER_PID" 2>/dev/null || kill -9 "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
    sleep 8
}
trap stop_server EXIT
trap 'stop_server; exit 130' INT TERM HUP

# Wait for readiness, but verify the responder is OUR process. A previous round
# recorded a clean run against another arm's leftover server because the health
# check only checked for HTTP 200.
wait_ready() {
    local arm="$1"
    for i in $(seq 900); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "  !! $arm: server died during startup (see ${arm}.server.log)"
            return 1
        fi
        if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null; then
            log "  $arm ready after ${i}s on port $PORT"
            return 0
        fi
        sleep 1
    done
    log "  !! $arm: timed out waiting for readiness"
    return 1
}

# Refuse to start an arm whose port is already answering. Every corrupted round in
# this project traced back to a live server nobody accounted for; this makes that
# state abort the arm instead of silently becoming its baseline.
port_is_free() {
    curl -sf -m 3 "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null && return 1
    return 0
}

ARM_INDEX=0

run_arm() {
    local arm="$1"; shift
    PORT=$(( PORT_BASE + ARM_INDEX )); ARM_INDEX=$(( ARM_INDEX + 1 ))
    if ! port_is_free; then
        log "!! $arm: port $PORT is already serving -- refusing to run this arm"
        return 1
    fi
    local prog="$1"; shift          # `vllm` or `unieinfra` -- argv[0] gates our defaults
    log "=== arm: $arm ($prog $*) ==="

    # argv[0] gates the defaults layer, but NOT the kernel selection layer, which is
    # on by default in this tree. The three vllm-launched arms are meant to isolate a
    # single lever each against untouched upstream, so the kernel layer is off for all
    # of them; only the unieinfra arm gets the full stack.
    local gemm=0; [ "$prog" = "unieinfra" ] && gemm=1

    setsid env CUDA_VISIBLE_DEVICES="$GPU" VLLM_DISABLE_COMPILE_CACHE=1 UNIEINFRA_GEMM="$gemm" \
        "$prog" serve "$MODEL" --port "$PORT" \
        --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs "$MAX_NUM_SEQS" \
        "$@" > "$OUT/${arm}.server.log" 2>&1 &
    SERVER_PID=$!
    log "  launched pid $SERVER_PID"

    if ! wait_ready "$arm"; then stop_server; return 1; fi

    # Record what the server actually decided, not what we asked for.
    grep -iE "kv cache|KV_CACHE|speculative|prefix cach|Unie|GPU KV cache size|maximum concurrency" \
        "$OUT/${arm}.server.log" | head -40 > "$OUT/${arm}.config.txt"

    for c in $CONCURRENCIES; do
        local n=$(( c * 8 )); [ "$n" -lt 32 ] && n=32; [ "$n" -gt 512 ] && n=512
        log "  bench c=$c n=$n"
        vllm bench serve \
            --backend vllm --model "$MODEL" --port "$PORT" \
            --dataset-name random --random-input-len "$ILEN" --random-output-len "$OLEN" \
            --num-prompts "$n" --max-concurrency "$c" \
            --ignore-eos --seed 0 --percentile-metrics ttft,tpot,itl,e2el \
            --save-result --result-dir "$OUT" \
            --result-filename "${arm}_c${c}.json" \
            >> "$OUT/${arm}.client.log" 2>&1 \
            || log "  !! bench failed at c=$c (continuing)"
    done

    stop_server
    log "=== arm $arm done ==="
}

log "########## E51 start: model=$MODEL gpu=$GPU ilen=$ILEN olen=$OLEN ##########"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tee -a "$OUT/driver.log"

# ARMS selects which arms to run, so a follow-up sweep can re-measure one lever
# without paying for the others. SPEC_K sets the fixed depth of the `mtp` arm.
ARMS="${ARMS:-stock mtp fp8kv unieinfra}"
SPEC_K="${SPEC_K:-3}"
for arm in $ARMS; do
    case "$arm" in
        stock)     run_arm stock     vllm ;;
        mtp)       run_arm mtp       vllm --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K}" ;;
        fp8kv)     run_arm fp8kv     vllm --kv-cache-dtype fp8 ;;
        unieinfra) run_arm unieinfra unieinfra ;;
        *) log "unknown arm: $arm" ;;
    esac
done

log "########## E51 complete ##########"
