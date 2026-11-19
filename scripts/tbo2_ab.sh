#!/bin/bash
# AgentX STRESS 600 s, prefill TBO off/on/on/off at one concurrency (repeat_ab.sh G-arm env,
# engine from the .wt-tbo2 worktree). Takes logs/GPU_LOCK for the batch.
# Usage: tbo2_ab.sh <conc>
cd /home/smc/workspace/roy || exit 1
C=${1:?conc}; M=logs/tbo2_ab.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=tbo2-agent(tbo2_ab c$C) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {  # tbo rep
  local t=$1 r=$2
  echo "$(date -u +%T) start TBO$t c$C r$r" >> $M
  env STRESS=1 TOKENSPEED_PREFILL_TBO=$t TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 \
    TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
    ENGINE_PYTHONPATH=/workspace/.wt-tbo2/python \
    TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*C)) \
    MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 \
    EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
    scripts/run_round.sh TBO2T$t-c$C-r$r 600 unieinfra "$C"
  echo "$(date -u +%T) done TBO$t c$C r$r" >> $M
}
run 0 1; run 1 1; run 1 2; run 0 2
echo "TBO2 AB c$C DONE $(date -u +%FT%TZ)" >> $M
