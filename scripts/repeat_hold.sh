#!/bin/bash
# Repeated AgentX stress (600 s) at c16/c32/c64: G = gluon MoE + BV (PR build), A = aiter MoE (a5ae4510) + BV.
# Order G A A G per concurrency so drift cancels.
cd /home/smc/workspace/roy; M=logs/repeat_hold.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(repeat_hold) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {  # kind c rep
  local k=$1 c=$2 r=$3 extra
  if [ $k = G ]; then extra="ENGINE_PYTHONPATH=/workspace/.wt-hostfix/python TOKENSPEED_HOLD_RESULTS_UNTIL_LAUNCH=0"
  else extra="ENGINE_PYTHONPATH=/workspace/.wt-hostfix/python TOKENSPEED_HOLD_RESULTS_UNTIL_LAUNCH=1"; fi
  echo "$(date -u +%T) start $k c$c r$r" >> $M
  env STRESS=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 $extra \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*c)) MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh RH$k$c$r 600 unieinfra "$c"
  echo "$(date -u +%T) done $k c$c r$r" >> $M
}
for c in 16 32 64; do run G $c 1; run A $c 1; run A $c 2; run G $c 2; done
echo "REPEAT HOLD DONE $(date -u +%FT%TZ)" >> $M
