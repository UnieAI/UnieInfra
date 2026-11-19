#!/bin/bash
# AgentX stress 600 s with prefill TBO on (prefill-tbo2 6361be94) + block verify; fills c1/c4/c64/c128 (c16/c32 from tbo2_ab).
cd /home/smc/workspace/roy; M=logs/stress_tbo.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress_tbo) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {
  echo "$(date -u +%T) start c$1 r$2" >> $M
  STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-tbo2/python TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*$1)) \
  MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh STB$1r$2 600 unieinfra "$1"
  echo "$(date -u +%T) done c$1 r$2" >> $M
}
run 64 1; run 4 1; run 128 1; run 1 1; run 64 2; run 4 2
echo "STRESS TBO DONE $(date -u +%FT%TZ)" >> $M
