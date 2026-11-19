#!/bin/bash
# c4–c64 wave screen: BV + prefill TBO (6361be94) plus TBO_MIN_ROWS=1024 (hide the 1K–4K-row prefill all-reduces)
# and Iris small-message (32 program) cap raised 2 MB -> 8 MB (decode 384–768 rows). Compare vs STB/TBO2T1 cells.
cd /home/smc/workspace/roy; M=logs/stress_e2.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress_e2) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {
  echo "$(date -u +%T) start c$1 r$2" >> $M
  STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_PREFILL_TBO_MIN_ROWS=${MINR:-1024} TOKENSPEED_IRIS_PULL_SMALL_MAX_BYTES=${SMALLB:-8388608} \
  TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-wave/python TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*$1)) \
  MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh STE2_$1r$2 600 unieinfra "$1"
  echo "$(date -u +%T) done c$1 r$2" >> $M
}
run 64 3; run 32 1; run 64 4; run 32 2
echo "STRESS E2B DONE $(date -u +%FT%TZ)" >> $M
