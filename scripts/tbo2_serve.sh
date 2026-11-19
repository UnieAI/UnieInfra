#!/bin/bash
# Start unieinfra with the repeat_ab.sh G-arm env from the prefill-TBO worktree.
# Usage: tbo2_serve.sh <0|1> <conc>
cd /home/smc/workspace/roy || exit 1
TBO=${1:?tbo 0|1}; C=${2:-32}
env TOKENSPEED_PREFILL_TBO=$TBO TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 \
  TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-tbo2/python \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*C)) \
  MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/serve_unieinfra.sh "$C"
