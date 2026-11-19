#!/bin/bash
# BV shard-local TP verify (c4c64-wave 49c86553) on top of BV + prefill TBO; chunk 64; screen c32/c64 (+c128 with MAX_BS 256).
cd /home/smc/workspace/roy; M=logs/stress_bvtp.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress_bvtp) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {  # c rep [extra env]
  echo "$(date -u +%T) start c$1 r$2" >> $M
  env STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 $3 \
  ENGINE_PYTHONPATH=/workspace/.wt-wave/python TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*$1)) \
  MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=${GMU:-0.9} EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh SBT$1r$2 600 unieinfra "$1"
  echo "$(date -u +%T) done c$1 r$2" >> $M
}
run 32 1; run 64 1; run 32 2; run 64 2
echo "STRESS BVTP DONE $(date -u +%FT%TZ)" >> $M
