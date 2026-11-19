#!/bin/bash
# c4–c64 wave screen: BV + prefill TBO (6361be94) plus TBO_MIN_ROWS=1024 (hide the 1K–4K-row prefill all-reduces)
# and Iris small-message (32 program) cap raised 2 MB -> 8 MB (decode 384–768 rows). Compare vs STB/TBO2T1 cells.
cd /home/smc/workspace/roy; M=logs/stress_e12.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress_e12) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {
  echo "$(date -u +%T) start c$1 r$2" >> $M
  STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_PREFILL_TBO_MIN_ROWS=${MINR:-1024} TOKENSPEED_IRIS_PULL_SMALL_MAX_BYTES=${SMALLB:-8388608} \
  TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-tbo2/python TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*$1)) \
  MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh STE$1r$2 600 unieinfra "$1"
  echo "$(date -u +%T) done c$1 r$2" >> $M
}
run 32 1; run 64 1; run 32 2; run 64 2
echo "STRESS E12 DONE $(date -u +%FT%TZ)" >> $M
