#!/bin/bash
# SGLang ROCm server for DeepSeek-V4.1-Flash-1017 on 4x MI350P (TP4).
# Based on InferenceX dsv41flash/sglang/mi355x-fp4-mtp/agentic.yaml (TP4 overrides).
# MI350P changes: Engram host table is already on in the recipe; the static memory
# fraction is raised because the recipe's 0.70 was sized for 288 GB MI355X.
# --monokernel-backend amd is dropped: its launch grid needs 256 CUs, MI350P has 128.
# Usage: serve_sglang.sh <CONC>
set -euo pipefail
CONC=${1:?usage: serve_sglang.sh <CONC>}
PORT=${PORT:-8000}
IMAGE=lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021
NAME=bench-server
MEM_FRACTION=${MEM_FRACTION:-0.85}

# Per-concurrency knobs from the recipe's override_tp4_c<N> blocks.
case "$CONC" in
  1)  PER_CONC=(--prefill-decode-interval 192 --schedule-policy lpm --swa-prefix-tails 128  --max-running-requests 2) ;;
  2)  PER_CONC=(--prefill-decode-interval 192 --schedule-policy lpm --swa-prefix-tails 128  --max-running-requests 4) ;;
  4)  PER_CONC=(--prefill-decode-interval 192 --schedule-policy lpm --swa-prefix-tails 256  --max-running-requests 8) ;;
  8)  PER_CONC=(--prefill-decode-interval 192 --schedule-policy lpm --swa-prefix-tails 512  --max-running-requests 16) ;;
  16) PER_CONC=(--prefill-decode-interval 96  --swa-prefix-tails 1024 --max-running-requests 32) ;;
  # c32+: the cookbook's MI355X agentic cells (c32: tails 2048 / 64 running;
  # c64: tails 4096) need 288 GB GPUs; on 144 GB MI350P their SWA pool cap
  # leaves no room for the full KV pool and the server refuses to start.
  # Keep the cookbook's interval/policy, shrink tails, cap running at 64.
  32)     PER_CONC=(--prefill-decode-interval 32 --schedule-policy fcfs --swa-prefix-tails 1024 --max-running-requests 64) ;;
  64|128) PER_CONC=(--prefill-decode-interval 16 --schedule-policy fcfs --swa-prefix-tails 512  --max-running-requests 64) ;;
  *)  echo "no recipe knobs for CONC=$CONC" >&2; exit 1 ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add 993 \
  --security-opt seccomp=unconfined --network host --ipc host --shm-size 64g --ulimit core=0 \
  -v /data/models:/models:ro \
  -e HF_HUB_OFFLINE=1 -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -e SGLANG_TIMEOUT_KEEP_ALIVE=900 -e SGLANG_DEFAULT_THINKING=1 \
  -e SGLANG_DSV41_REASONING_EFFORT=high \
  -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 -e SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
  -e SGLANG_USE_AITER=1 -e SGLANG_USE_ROCM700A=0 -e TRITON_HIP_USE_ASYNC_COPY=0 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 -e SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true \
  -e SGLANG_OPT_HIP_FLYDSL_MXFP8=1 -e SGLANG_HIP_FP4_INDEXER_ROWGROUP=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 -e HSA_NO_SCRATCH_RECLAIM=0 -e GPU_MAX_HW_QUEUES=2 \
  --entrypoint python3 "$IMAGE" -m sglang.launch_server \
  --model-path /models/DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --trust-remote-code --tensor-parallel-size 4 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --enforce-shared-experts-fusion --fp8-gemm-backend aiter \
  --cuda-graph-max-bs-decode 64 --enable-decoder-swa-bounded-replay \
  --cuda-graph-backend-prefill disabled \
  --reasoning-parser auto --tool-call-parser auto \
  --watchdog-timeout 3600 --enable-metrics \
  --mem-fraction-static "$MEM_FRACTION" --chunked-prefill-size 16384 \
  "${PER_CONC[@]}"
echo "started $NAME (sglang, CONC=$CONC) on port $PORT"
