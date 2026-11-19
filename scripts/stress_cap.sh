#!/bin/bash
# AgentX stress 600 s, mi350p-final: raise MAX_NUM_SEQS above 2x c (subagents push concurrent requests to ~22 at c4).
cd /home/smc/workspace/roy; M=logs/stress_cap.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress_cap) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {
  echo "$(date -u +%T) start c$1 $2" >> $M
  STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-final/tokenspeed-kernel-amd/python:/workspace/.wt-final/tokenspeed-kernel/python:/workspace/.wt-final/python \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$3 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh SCP$1$2 600 unieinfra "$1"
  echo "$(date -u +%T) done c$1 $2" >> $M
}
run 4 c16 16; run 4 c32 32; run 16 c64 64
echo "STRESS CAP DONE $(date -u +%FT%TZ)" >> $M
