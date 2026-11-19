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
docker exec -u root ts-dev bash -c '
  P="smg_grpc_servicer|tokenspeed::|tokenspeed serve|deepseek-v4.1-flash-1017.sh"
  pkill -f "$P"; sleep 8; pkill -9 -f "$P"; sleep 2' || true
for _ in $(seq 1 30); do
  used=$(amd-smi metric --mem-usage 2>/dev/null | awk "/USED_VRAM/ {s+=\$2} END {print s+0}")
  (( used < 8000 )) && break
  sleep 2
done

# AgentX lanes fan out into subagents, so allow 2x CONC running requests
# (same as the SGLang recipe's max-running-requests).
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$(( CONC < 1 ? 2 : 2 * CONC ))}
# DSpark verifies 5 drafts + 1 token per request per decode step; cap the
# decode CUDA-graph ladder at 512 tokens so c64/c128 graphs don't eat KV memory.
CAPTURE=$(( MAX_NUM_SEQS * 6 ))
(( CAPTURE > 512 )) && CAPTURE=512
# KV budget in tokens (all host-side so env overrides reach the container).
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-16777216}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
docker exec -d -u root \
  -e MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017 \
  -e SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  -e HOST_PORT="$PORT" -e TENSOR_PARALLEL_SIZE=4 -e CUDA_DEVICES=0,1,2,3 \
  -e ENGRAM_HOST_TABLE=${ENGRAM_HOST_TABLE:-1} -e LANGUAGE_MODEL_ONLY=1 \
  -e MAX_MODEL_LEN=1048576 -e MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}" \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="$CAPTURE" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  -e EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}" \
  -e HF_HUB_OFFLINE=1 -e PYTHONFAULTHANDLER=1 -e TORCH_SHOW_CPP_STACKTRACES=1 \
  -e AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-0}" -e HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-0}" \
  -e HIP_FORCE_DEV_KERNARG -e GPU_MAX_HW_QUEUES -e DEBUG_CLR_GRAPH_PACKET_CAPTURE \
  -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1} -e TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS -e TS_EXP_DECODE_INDEX_QUERY_TILE -e TOKENSPEED_DATA_PLANE_SYNC_DEBUG -e TS_EXP_TIME_BT -e TS_EXP_ZERO_ON_EXEC -e TOKENSPEED_IRIS_PULL_MAX_PROGRAMS -e TOKENSPEED_IRIS_PULL_SMALL_PROGRAMS -e TOKENSPEED_IRIS_PULL_SMALL_MAX_BYTES -e TOKENSPEED_V41_TAIL_ROWS -e TOKENSPEED_DSPARK_VERIFY_DRAFTS -e TOKENSPEED_CACHE_AWARE_ADMISSION -e TOKENSPEED_CACHE_AWARE_MAX_OVERTAKES -e TOKENSPEED_MIXED_REF_CHUNK -e TOKENSPEED_MIXED_DECODE_REF -e TOKENSPEED_MIXED_LONG_CHUNK -e TOKENSPEED_MIXED_PREFILL_FLOOR -e TS_EXP_IRIS_AR_MAX_BYTES -e TS_EXP_IRIS_PD_MAX_BYTES -e TS_EXP_FFN_OVERLAP_CHUNKS -e TS_EXP_FFN_OVERLAP_MIN_ROWS -e TS_EXP_FFN_OVERLAP_IRIS_PROGRAMS \
  ts-dev bash -c "ulimit -c 0; cd /workspace && exec env LD_LIBRARY_PATH=/opt/rocm/lib LD_PRELOAD=librocprofiler-register.so:librocprofiler-sdk.so:libhsa-amd-aqlprofile64.so rocprofv3 --pmc TCC_EA0_RDREQ_128B_sum TCC_EA0_RDREQ_64B_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_WRREQ_64B_sum -d /logs/hbmprof -o r --output-format csv -- ./scripts/deepseek-v4.1-flash-1017.sh > $LOG 2>&1"
ln -sfn "$LOG_NAME" /home/smc/workspace/roy/logs/unieinfra-server.log
echo "started unieinfra (CONC=$CONC) on port $PORT, log: /home/smc/workspace/roy/logs/$LOG_NAME"
