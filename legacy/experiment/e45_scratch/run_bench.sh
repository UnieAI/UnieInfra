#!/usr/bin/env bash
# Wait for an idle GPU (never GPU 0) then run the E45 bench inside the container.
set -eo pipefail
REPO=/workspace/roy/UnieInfra
THRESH=${THRESH:-3000}
GPU=""
for _ in $(seq 1 480); do
  for i in 3 2 1; do
    used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -lt "$THRESH" ]; then GPU=$i; break; fi
  done
  [ -n "$GPU" ] && break
  sleep 15
done
if [ -z "$GPU" ]; then echo "NO FREE GPU"; exit 1; fi
echo "[e45] GPU $GPU  args: $*"
exec docker exec -e CUDA_VISIBLE_DEVICES=$GPU -e TRITON_CACHE_DIR=/tmp/e45_triton \
  unieinfra-vllm-build bash -lc \
  "source /workspace/roy/venvs/unieinfra-vllm-venv/bin/activate; cd $REPO && \
   python benchmark/scripts/bench_decode_batch_gemm.py $*"
