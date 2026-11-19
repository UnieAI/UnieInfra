#!/bin/bash
# DSV4.1 TP4 on MI350P with Host L2 + file L3 (PR #49), DSpark off (KVStore needs it off).
# Engine code: /workspace/.wt-l3 (branch l3-file-backend). Usage: serve_l3.sh <tag>
set -euo pipefail
TAG=${1:?tag}
LOG_NAME=unieinfra-l3-$TAG-$(date -u +%Y%m%dT%H%M%SZ).log
docker exec -u root ts-dev bash -c '
  P="smg[_]grpc_servicer|smg[:][:]|tokenspeed[:][:]|tokenspeed[ ]serve|deepseek-v4[.]1-flash-1017[.]sh|ds[_]l3[.]sh|multiprocessing[.]spawn"
  pkill -f "$P"; sleep 6; pkill -9 -f "$P"; sleep 2' || true
for i in $(seq 60); do
  u=$(amd-smi metric --mem-usage 2>/dev/null | awk '/USED_VRAM/ {s+=$2} END {print s+0}')
  (( u < 8000 )) && break; sleep 2
done
L3CFG="{\"path\":\"/workspace/.l3cache\",\"capacity_gb\":60,\"io_threads\":${IO_THREADS:-4},\"o_direct\":true,\"verify_crc\":true,\"fsync\":false}"
EXTRA="--kvstore-size ${KVSTORE_GB:-4} --kvstore-storage-backend file --kvstore-storage-backend-extra-config $L3CFG --kvstore-prefetch-min-pages 1 --kvstore-prefetch-timeout-base-s 10 --kvstore-prefetch-timeout-per-page-s 0.05 --kvstore-prefetch-batch-pages 128 ${EXTRA_SERVE_ARGS:-}"
docker exec -d -u root \
  -e PYTHONPATH=/workspace/${ENGINE_WT:-.wt-l3}/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor \
  -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -e MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017 -e SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  -e HOST_PORT=8000 -e TENSOR_PARALLEL_SIZE=4 -e CUDA_DEVICES=0,1,2,3 \
  -e ENGRAM_HOST_TABLE=1 -e LANGUAGE_MODEL_ONLY=1 -e ENABLE_DSPARK=${ENABLE_DSPARK:-0} -e TOKENSPEED_KVSTORE_WRITE_POLICY=${WRITE_POLICY:-eager} \
  -e MAX_MODEL_LEN=262144 -e MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-262144} \
  -e MAX_NUM_SEQS=8 -e CHUNKED_PREFILL_SIZE=4096 -e MAX_CUDAGRAPH_CAPTURE_SIZE=16 \
  -e GPU_MEMORY_UTILIZATION=0.90 -e EXTRA_SERVE_ARGS="$EXTRA" \
  -e HF_HUB_OFFLINE=1 -e PYTHONFAULTHANDLER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  ts-dev bash -c "ulimit -c 0; cd /workspace && exec ./.l3bench/ds_l3.sh > /logs/$LOG_NAME 2>&1"
echo /home/smc/workspace/roy/logs/$LOG_NAME
