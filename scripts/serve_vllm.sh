#!/bin/bash
# vLLM ROCm server for DeepSeek-V4.1-Flash-1017 on 4x MI350P (TP4).
# Based on InferenceX dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml (TP4 overrides).
# MI350P change: 144 GB HBM/GPU (vs 288 GB on MI355X), so the Engram tables are
# offloaded to host memory (engram-config cpu_offload=true). --monokernel is
# dropped: vLLM mono decode requires 256 CUs and MI350P has 128.
# Usage: serve_vllm.sh <CONC>   (selects the per-concurrency knobs from the recipe)
set -euo pipefail
CONC=${1:?usage: serve_vllm.sh <CONC>}
PORT=${PORT:-8000}
IMAGE=vllm/vllm-openai-rocm@sha256:3c634cc2c51eda7eb840a236b1f09a5ecf6a4b0cb92c4b8486e2158770507291
NAME=bench-server

# Per-concurrency knobs copied verbatim from the recipe's override_tp4_c<N> blocks.
case "$CONC" in
  1|2) PER_CONC=(--max-cudagraph-capture-size 8 --max-num-batched-tokens 32768 --max-num-seqs 1) ;;
  # MI350P: the recipe leaves max-num-seqs at the default for c4/c8, whose
  # buffers leave only 2.27 GiB KV on 144 GB (< 2.41 GiB for one 1M request).
  # Cap it at 2x CONC, as for SGLang/unieinfra.
  4|8) PER_CONC=(--max-cudagraph-capture-size 1024 --max-num-batched-tokens 16384 --max-num-seqs $(( 2 * CONC ))) ;;
  16)  PER_CONC=(--max-cudagraph-capture-size 1024 --max-num-batched-tokens 8192 --max-num-seqs 4) ;;
  32)  PER_CONC=(--max-cudagraph-capture-size 1024 --max-num-batched-tokens 8192 --max-num-seqs 8) ;;
  # The recipe stops at c32 (max-num-seqs = CONC/4); c64/c128 extend that ratio.
  64|128|256) PER_CONC=(--max-cudagraph-capture-size 1024 --max-num-batched-tokens 8192 --max-num-seqs $(( CONC / 4 ))) ;;
  *)   echo "no recipe knobs for CONC=$CONC" >&2; exit 1 ;;
esac

# Algorithm-aligned baseline (VLLM_ALIGN=1): the same speculative algorithm as the
# UnieInfra baseline (greedy draft, standard rejection: the draft distribution is
# one-hot, so a draft token is accepted with the target's probability of it) and one
# token budget for every concurrency. Launch settings only; vLLM's code is untouched.
SPEC_CFG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
if [[ "${VLLM_ALIGN:-0}" == 1 ]]; then
  SPEC_CFG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
  for i in "${!PER_CONC[@]}"; do
    if [[ "${PER_CONC[$i]}" == "--max-num-batched-tokens" ]]; then
      PER_CONC[$((i + 1))]=${VLLM_ALIGN_BATCHED_TOKENS:-8192}
    fi
  done
fi

# VLLM_MAX_NUM_SEQS overrides the recipe's max-num-seqs (sensitivity runs only:
# the recipe caps c64/c128 at CONC/4 running requests).
if [[ -n "${VLLM_MAX_NUM_SEQS:-}" ]]; then
  PER_CONC+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
# The reference recipe runs the sparse indexer with an mxfp4 key cache
# (indexer_kv_dtype) and mxfp4 sparse logits. UnieInfra has no equivalent without
# quantizing, and the goal forbids quantization, so the level playing field is a
# vLLM reference without it: VLLM_NO_INDEXER_QUANT=1 drops the whole
# --attention-config override and takes the engine's own default dtypes.
if [[ "${VLLM_NO_INDEXER_QUANT:-0}" == "1" ]]; then
  ATTN_CFG=()
else
  ATTN_CFG=(--attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}')
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add 993 \
  --security-opt seccomp=unconfined --network host --ipc host --shm-size 64g --ulimit core=0 \
  -v /data/models:/models:ro -v /home/smc/workspace/roy/logs:/logs \
  ${VLLM_TORCH_PROFILER_DIR:+-e VLLM_TORCH_PROFILER_DIR=$VLLM_TORCH_PROFILER_DIR} \
  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE -e AITER_TRITON_LOG_LEVEL=ERROR \
  -e VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e OMP_NUM_THREADS=1 -e VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-1} -e PYTHONUNBUFFERED=1 \
  --entrypoint vllm "$IMAGE" serve /models/DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --tensor-parallel-size 4 \
  --language-model-only \
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --moe-backend aiter --gpu-memory-utilization 0.9 \
  "${ATTN_CFG[@]}" \
  --block-size 128 \
  --speculative-config "$SPEC_CFG" \
  --max-model-len 1048576 --disable-uvicorn-access-log --enable-prompt-tokens-details \
  --engram-config '{"cpu_offload":true}' \
  "${PER_CONC[@]}" ${VLLM_PROFILER_CONFIG:+--profiler-config "$VLLM_PROFILER_CONFIG"}
# Keep the container's log after the container is gone (the KV cache size it chose is read from it).
setsid nohup docker logs -f "$NAME" > /home/smc/workspace/roy/logs/vllm-last-server.log 2>&1 &
echo "started $NAME (vllm, CONC=$CONC) on port $PORT"
