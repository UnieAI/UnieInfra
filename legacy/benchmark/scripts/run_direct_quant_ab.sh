#!/usr/bin/env bash
# UnieInfra -- fp8 / NVFP4 A/B on the DIRECT seam (unieinfra-vllm), not the plugin.
#
#   ./run_direct_quant_ab.sh fp8   /workspace/models/Qwen/Qwen3-8B
#   GPU="2 3" ./run_direct_quant_ab.sh nvfp4 /workspace/models/nvidia/Qwen3-8B-NVFP4
#
# GPU is a candidate LIST, re-picked before every arm; the runner adopts
# whichever card is completely idle. GPU 0 is deliberately not a default
# candidate -- it belongs to another user.
#
# Runs INSIDE the `unieinfra-vllm-build` container, which is where the fork is
# installed editable and where the compiled extensions match the torch build.
#
# Arms (one process each, rotated, `ROUNDS` times):
#
#   vllm    the engine's own quantized path                    UNIEINFRA_QUANT=0
#   ours    UnieInfra dispatch, activation quant HOISTED out   MODE=hoist
#   opaque  UnieInfra dispatch, E38's shape (quant inside)     MODE=opaque
#   ctrl    kernel UNREGISTERED, so no layer is ever claimed -- must land on
#           `vllm` to 0.0%, which is what makes the attribution readable (E38)
#   tax     exactly the layers `ours` claims, but the registry is never asked --
#           every call falls through to the engine kernel from inside the
#           dispatch op. `vllm - tax` is therefore the cost of the OP ITSELF,
#           which E38's control could not isolate: unregistering the kernel
#           also un-claims the layer, so the op vanishes with it.
#   taxh    the same, in `hoist` mode: how much of that cost the visible
#           activation quantization gives back.
#
# `UNIEINFRA_GEMM=0` and `UNIEINFRA_ATTENTION=0` on EVERY arm, so the quantized
# dispatch is the only variable. `VLLM_DISABLE_COMPILE_CACHE=1` because vLLM's
# compile cache keys on VllmConfig, not on env vars, and these arms differ only
# in env (E34 trap (c)).
set -euo pipefail

MODE="${1:?usage: run_direct_quant_ab.sh <fp8|nvfp4> <model>}"
MODEL="${2:?usage: run_direct_quant_ab.sh <fp8|nvfp4> <model>}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/workspace/roy/UnieInfra}"
CONTAINER="${CONTAINER:-unieinfra-vllm-build}"
VENV="${VENV:-/workspace/roy/venvs/unieinfra-vllm-venv}"
GPU="${GPU:-1 2 3}"   # candidate list, re-picked before every arm
ROUNDS="${ROUNDS:-2}"
OUTDIR="${OUTDIR:-$REPO/benchmark/results/vllm-direct-quant}"
ARMS="${ARMS:-vllm ours opaque ctrl tax taxh}"
mkdir -p "$OUTDIR/raw" "$OUTDIR/selection" "$OUTDIR/logs"

if [ "$MODE" = "nvfp4" ]; then
    # sm_120 shims, on BOTH arms: vLLM's `auto` NVFP4 backend picks FlashInfer,
    # whose JIT rejects `major == 12`, and the checkpoint's fp8 KV-cache algo
    # drags in the FlashInfer attention backend for the same reason.
    EXTRA="BENCH_LINEAR_BACKEND=cutlass BENCH_ATTN=TRITON_ATTN"
    NATIVE_QUANT=modelopt_fp4
    DISABLE_VAR=UNIEINFRA_DECODE_GEMV_NVFP4
else
    EXTRA=""
    NATIVE_QUANT=fp8
    DISABLE_VAR=UNIEINFRA_DECODE_GEMV_FP8
fi

arm_env() {
    case "$1" in
    vllm) echo "UNIEINFRA_QUANT=0" ;;
    ours) echo "UNIEINFRA_QUANT=1 UNIEINFRA_QUANT_MODE=hoist" ;;
    opaque) echo "UNIEINFRA_QUANT=1 UNIEINFRA_QUANT_MODE=opaque" ;;
    ctrl) echo "UNIEINFRA_QUANT=1 UNIEINFRA_QUANT_MODE=opaque ${DISABLE_VAR}=0" ;;
    tax) echo "UNIEINFRA_QUANT=1 UNIEINFRA_QUANT_MODE=opaque UNIEINFRA_QUANT_NEVER_SELECT=1" ;;
    taxh) echo "UNIEINFRA_QUANT=1 UNIEINFRA_QUANT_MODE=hoist UNIEINFRA_QUANT_NEVER_SELECT=1" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

# `GPU` is a candidate list, not a single device: five agents share this box and
# which card is idle changes minute to minute. GPU 0 is never a candidate --
# it belongs to another user.
GPU_CANDIDATES="${GPU:-1 2 3}"
GPU=""
GPU_UUID=""

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
gpu_procs() {
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader |
        grep -c "$1" || true
}

# Wait for *some* candidate GPU to be completely idle, and adopt it. A number
# taken next to someone else's benchmark is not a number.
acquire_gpu() {
    for _ in $(seq 1 480); do
        for g in $GPU_CANDIDATES; do
            local uuid
            uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g")"
            if [ "$(gpu_used "$g")" -lt 2000 ] && [ "$(gpu_procs "$uuid")" -eq 0 ]; then
                GPU="$g"
                GPU_UUID="$uuid"
                return 0
            fi
        done
        sleep 15
    done
    echo "no candidate GPU went quiet ($GPU_CANDIDATES)" >&2
    return 1
}

# Sample GPU memory for the duration of an arm. Contention is judged on MEMORY,
# not on the process count: an arm is TWO processes of ours (the offline `LLM`
# driver briefly attaches a ~550 MiB context of its own alongside the ~59 GiB
# EngineCore), so counting processes rejects every arm on the strength of our
# own driver. Our own ceiling is `BENCH_GPU_UTIL x total` plus that driver, so
# anything meaningfully above it is somebody else.
watch_contention() {
    local out="$1" dev="$2"
    : >"$out"
    while :; do
        gpu_used "$dev" >>"$out"
        sleep 5
    done
}

# MiB above which the GPU is holding more than this arm possibly can.
contention_ceiling() {
    local total util
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU")
    util="${BENCH_GPU_UTIL:-0.55}"
    awk -v t="$total" -v u="$util" 'BEGIN { printf "%d", t * u + 8000 }'
}

run() {
    local arm="$1" rep="$2" label="${MODE}_${1}_r${2}"
    echo "=== $label ==="
    acquire_gpu || return 1
    echo "using GPU $GPU"
    local watchfile watchpid maxproc
    watchfile="$OUTDIR/logs/$label.contention"
    watch_contention "$watchfile" "$GPU" &
    watchpid=$!
    docker exec "$CONTAINER" bash -lc "
        source $VENV/bin/activate
        cd $REPO
        export CUDA_VISIBLE_DEVICES=$GPU
        export VLLM_DISABLE_COMPILE_CACHE=1
        export UNIEINFRA_GEMM=0
        export UNIEINFRA_ATTENTION=0
        export BENCH_MODEL='$MODEL'
        export BENCH_QUANT='$NATIVE_QUANT'
        export BENCH_CONC='${BENCH_CONC:-1,32,128}'
        export BENCH_OLEN='${BENCH_OLEN:-256}'
        export BENCH_ILEN='${BENCH_ILEN:-512}'
        export BENCH_REPS='${BENCH_REPS:-3}'
        export BENCH_GPU_UTIL='${BENCH_GPU_UTIL:-0.55}'
        export BENCH_TAG='$label'
        export BENCH_OUT='$OUTDIR/raw/$label.json'
        export UNIEINFRA_KERNEL_SELECTION_DUMP='$OUTDIR/selection/$label'
        export $EXTRA $(arm_env "$arm")
        python benchmark/scripts/bench_quant_arm.py
    " >"$OUTDIR/logs/$label.log" 2>&1 || {
        kill "$watchpid" 2>/dev/null
        echo "ARM FAILED: $label"
        tail -40 "$OUTDIR/logs/$label.log"
        return 1
    }
    kill "$watchpid" 2>/dev/null
    local ceiling
    ceiling=$(contention_ceiling)
    maxproc=$(sort -n "$watchfile" | tail -1)
    if [ "${maxproc:-0}" -gt "$ceiling" ]; then
        echo "ARM CONTENDED: $label peaked at ${maxproc} MiB on GPU $GPU" \
             "(ceiling ${ceiling}) -- discarding"
        rm -f "$OUTDIR/raw/$label.json"
        return 1
    fi
    echo "clean: peak ${maxproc} MiB on GPU $GPU (ceiling ${ceiling})"
    grep -E "MEDIAN" "$OUTDIR/logs/$label.log" || true
}

for r in $(seq 1 "$ROUNDS"); do
    for arm in $ARMS; do
        # Up to three attempts: on a shared box a contended arm is discarded,
        # not averaged in.
        run "$arm" "$r" || run "$arm" "$r" || run "$arm" "$r" ||
            echo "ARM GAVE UP: ${MODE}_${arm}_r${r}"
    done
done

echo
echo "=== summary ==="
"$HERE/summarize_direct_quant.py" "$OUTDIR" "$MODE"
