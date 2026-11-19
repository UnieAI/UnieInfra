#!/bin/bash
# unieinfra (TokenSpeed fork) server for DeepSeek-V4.1-Flash-1017 on 4x MI350P (TP4).
# Runs from the editable source checkout inside the long-lived ts-dev container
# (scripts/start_dev_container.sh), so source edits take effect on restart.
# Wraps the repo's own scripts/deepseek-v4.1-flash-1017.sh.
# MI350P change: 144 GB HBM/GPU, so the ~203 GB Engram tables go to host memory.
# Usage: serve_unieinfra.sh <CONC>      extra serve flags via EXTRA_SERVE_ARGS
set -euo pipefail
CONC=${1:?usage: serve_unieinfra.sh <CONC>}
PORT=${PORT:-8000}
# One log per launch (a later launch must not overwrite a crashed run's log);
# logs/unieinfra-server.log always points at the newest one.
LOG_NAME=unieinfra-server-c$CONC-$(date -u +%Y%m%dT%H%M%SZ).log
LOG=/logs/$LOG_NAME

# Stop any previous server in the dev container. The server runs as root and
# its workers are named smg_grpc_servicer / tokenspeed::scheduler_tp*, so stop
# them as root by those names, then wait until the GPUs are released.
kill_engine() {
  docker exec -u root ts-dev bash -c '
    # Bracketed so the pattern cannot match the bash -c command line that runs it
    # (a self-match kills the shell before the pkill -9 line, 2026-11-10).
    P="smg[_]grpc_servicer|smg[:][:]|tokenspeed[:][:]|tokenspeed[ ]serve|deepseek-v4[.]1-flash-1017[.]sh|multiprocessing[.]spawn|addr2line"
    pkill -f "$P"; sleep 6; pkill -9 -f "$P"; sleep 2
    stray=$(ps -eo pid,args --no-headers | grep -E "$P" | grep -v "grep -E" | awk "{print \$1}" | tr "\n" " ")
    [ -n "$stray" ] && { kill -9 $stray 2>/dev/null; sleep 3; }' || true
}
vram_used() { amd-smi metric --mem-usage 2>/dev/null | awk '/USED_VRAM/ {s+=$2} END {print s+0}'; }
wait_freed() {
  for ((i=0; i<$1; i++)); do (( $(vram_used) < 8000 )) && return 0; sleep 2; done
  return 1
}

# Launching into memory the previous cell has not released OOMs at RCCL init
# (2026-11-09 03:14: c64 cap=16 died 4 min in with GPU 3 still holding 426 GB,
# and the surviving scheduler then stalled the round for half an hour). Retry
# the kill once, then refuse to launch instead of walking into the same OOM.
kill_engine
if ! wait_freed 45; then
  echo "serve_unieinfra: $(vram_used) MB still held after kill -9, retrying" >&2
  kill_engine
  if ! wait_freed 15; then
    echo "serve_unieinfra: GPUs still hold $(vram_used) MB, refusing to start" >&2
    exit 1
  fi
fi

# AgentX lanes fan out into subagents, so allow 2x CONC running requests
# (same as the SGLang recipe's max-running-requests).
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$(( CONC < 1 ? 2 : 2 * CONC ))}
# DSpark verifies 5 drafts + 1 token per request per decode step; cap the
# decode CUDA-graph ladder at 512 tokens so c64/c128 graphs don't eat KV memory.
CAPTURE=$(( MAX_NUM_SEQS * 6 ))
(( CAPTURE > 512 )) && CAPTURE=512
# KV budget in tokens (all host-side so env overrides reach the container).
# 33.5M tokens (#81-#82): the old 16.7M cap, not memory, sized the KV pool; 47M+ OOMs at c64.
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-33554432}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
# INT8 two-shot for prefill-sized all-reduces (#93: c32 +12%, c128 +28%).
# DEFAULT OFF: comparisons against vLLM must be quantization-free, so the
# quantized comm path is opt-in only (a fair cell sets nothing here).
export TOKENSPEED_IRIS_Q8_ALLREDUCE=${TOKENSPEED_IRIS_Q8_ALLREDUCE:-0}
# Kernel worktree under test: /workspace is the main checkout, kept at HEAD, so
# an experimental kernel lives in a worktree and is prepended here. AITER comes
# from .vendor (the same wheels the reference image runs) and is only importable
# through this path.
ENGINE_PYTHONPATH=${ENGINE_PYTHONPATH:-/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor}
docker exec -d -u root \
  -e PYTHONPATH="$ENGINE_PYTHONPATH" -e LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib:/opt/rocm/lib}" \
  -e MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017 \
  -e SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  -e HOST_PORT="$PORT" -e TENSOR_PARALLEL_SIZE=4 -e CUDA_DEVICES=0,1,2,3 \
  -e ENGRAM_HOST_TABLE=1 -e LANGUAGE_MODEL_ONLY=1 \
  -e ENABLE_EXPERT_PARALLEL -e MOE_BACKEND \
  -e MAX_MODEL_LEN=1048576 -e MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}" \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="$CAPTURE" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  -e EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}" \
  -e HF_HUB_OFFLINE=1 -e PYTHONFAULTHANDLER=1 -e TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-0} \
  -e AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-0}" -e HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-0}" \
  -e HIP_FORCE_DEV_KERNARG -e GPU_MAX_HW_QUEUES -e DEBUG_CLR_GRAPH_PACKET_CAPTURE \
  -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1} -e TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS -e TOKENSPEED_PACK_FINISHING_EXTENDS -e TS_EXP_DECODE_INDEX_QUERY_TILE -e TOKENSPEED_DATA_PLANE_SYNC_DEBUG -e TS_EXP_TIME_BT -e TS_EXP_ZERO_ON_EXEC -e TOKENSPEED_IRIS_PULL_MAX_PROGRAMS -e TOKENSPEED_IRIS_PULL_SMALL_PROGRAMS -e TOKENSPEED_IRIS_PULL_SMALL_MAX_BYTES -e TOKENSPEED_V41_TAIL_ROWS -e TOKENSPEED_DSPARK_VERIFY_DRAFTS -e TOKENSPEED_IRIS_Q8_ALLREDUCE -e TOKENSPEED_IRIS_Q8_MIN_BYTES -e TOKENSPEED_IRIS_Q8_GROUP -e TOKENSPEED_SHADOW_NGRAM -e TOKENSPEED_SHADOW_NGRAM_N -e TOKENSPEED_SHADOW_NGRAM_K -e TOKENSPEED_SHADOW_NGRAM_PROMPT_TAIL -e TOKENSPEED_DSPARK_COUPLED_DRAFT -e TOKENSPEED_DSPARK_COUPLED_SHIFT -e TOKENSPEED_DSPARK_COUPLED_TEMP_SCALE -e ATTN_DP_SIZE -e TOKENSPEED_V41_KEEP_ALL_ROWS -e TOKENSPEED_CACHE_AWARE_ADMISSION -e TOKENSPEED_CACHE_AWARE_MAX_OVERTAKES -e TOKENSPEED_MIXED_REF_CHUNK -e TOKENSPEED_MIXED_DECODE_REF -e TOKENSPEED_MIXED_LONG_CHUNK -e TOKENSPEED_MIXED_PREFILL_FLOOR -e TS_EXP_IRIS_AR_MAX_BYTES -e TS_EXP_IRIS_PD_MAX_BYTES -e TS_EXP_FFN_OVERLAP_CHUNKS -e TS_EXP_FFN_OVERLAP_MIN_ROWS -e TS_EXP_FFN_OVERLAP_IRIS_PROGRAMS -e TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY -e TOKENSPEED_KERNEL_VERBOSE -e TOKENSPEED_DECODE_HOST_PREP -e TOKENSPEED_DSPARK_BLOCK_VERIFY -e TOKENSPEED_DSPARK_BLOCK_VERIFY_TP -e TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK -e TOKENSPEED_DSPARK_BLOCK_VERIFY_MAX_BS -e TOKENSPEED_GIL_SWITCH_INTERVAL_US -e TOKENSPEED_HOLD_RESULTS_UNTIL_LAUNCH -e TOKENSPEED_INDEX_RADIX_TOPK -e TOKENSPEED_MOE_DECODE_E4M3 -e TOKENSPEED_ATTN_ROWS_SPLIT -e TOKENSPEED_DRAFT_ATTN_FAST -e TOKENSPEED_DECODE_HC_INLINE -e DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING -e DEBUG_HIP_FORCE_GRAPH_QUEUES -e DEBUG_CLR_MAX_BATCH_SIZE -e DEBUG_CLR_BATCH_CPU_SYNC_SIZE -e DEBUG_HIP_GRAPH_BATCH_SIZE -e AMD_LOG_LEVEL -e TOKENSPEED_SCHED_EXP -e TOKENSPEED_GPU_STEP_TIMING -e AMD_LOG_MASK -e TOKENSPEED_DECODE_HOST_PREP_WAIT_MS -e TOKENSPEED_PREFILL_TBO -e TOKENSPEED_PREFILL_TBO_MIN_ROWS -e TOKENSPEED_PREFILL_TBO_IRIS_PROGRAMS -e TOKENSPEED_MOE_PREFILL_S1_V2 -e TOKENSPEED_MOE_PREFILL_S2_V2 -e TOKENSPEED_MOE_PREFILL_S2_REDUCE_V2 \
  ts-dev bash -c "ulimit -c 0; cd /workspace && exec ./scripts/deepseek-v4.1-flash-1017.sh > $LOG 2>&1"
ln -sfn "$LOG_NAME" /home/smc/workspace/roy/logs/unieinfra-server.log
echo "started unieinfra (CONC=$CONC) on port $PORT, log: /home/smc/workspace/roy/logs/$LOG_NAME"
