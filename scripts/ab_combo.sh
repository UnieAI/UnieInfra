#!/bin/bash
# A/B (A B B A) at c16/c32/c64, AgentX stress 600 s, gmu 0.86 for both arms.
# A = c4c64-wave 49c86553 (BV-TP + prefill TBO).  B = c4c64-combo d3a27814 (+ hold results until launch) + Iris small cap 8 MB.
cd /home/smc/workspace/roy; M=logs/ab_combo.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(ab_combo) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {  # arm c rep
  local k=$1 c=$2 r=$3 extra
  if [ $k = A ]; then extra="ENGINE_PYTHONPATH=/workspace/.wt-wave/python"
  else extra="ENGINE_PYTHONPATH=/workspace/.wt-combo2/python TOKENSPEED_HOLD_RESULTS_UNTIL_LAUNCH=1 TOKENSPEED_IRIS_PULL_SMALL_MAX_BYTES=8388608"; fi
  echo "$(date -u +%T) start $k c$c r$r" >> $M
  env STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 $extra \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*c)) MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh CB$k$c$r 600 unieinfra "$c"
  echo "$(date -u +%T) done $k c$c r$r" >> $M
}
for c in 32 64 16; do run A $c 1; run B $c 1; run B $c 2; run A $c 2; done
echo "AB COMBO DONE $(date -u +%FT%TZ)" >> $M
