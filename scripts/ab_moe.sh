#!/bin/bash
# A/B (A B B A) at c16/c32/c64, AgentX stress 600 s, gmu 0.86 for both arms.
# A = moe-xcd 62255315 flags off; B = same build + TOKENSPEED_MOE_PREFILL_S1_V2=1 S2_V2=1 (pipelined prefill MoE).
cd /home/smc/workspace/roy; M=logs/ab_moe.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(ab_moe) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {  # arm c rep
  local k=$1 c=$2 r=$3 extra
  if [ $k = A ]; then extra="ENGINE_PYTHONPATH=/workspace/.wt-moe/tokenspeed-kernel-amd/python:/workspace/.wt-moe/tokenspeed-kernel/python:/workspace/.wt-moe/python TOKENSPEED_MOE_PREFILL_S1_V2=0 TOKENSPEED_MOE_PREFILL_S2_V2=0"
  else extra="ENGINE_PYTHONPATH=/workspace/.wt-moe/tokenspeed-kernel-amd/python:/workspace/.wt-moe/tokenspeed-kernel/python:/workspace/.wt-moe/python TOKENSPEED_MOE_PREFILL_S1_V2=1 TOKENSPEED_MOE_PREFILL_S2_V2=1"; fi
  echo "$(date -u +%T) start $k c$c r$r" >> $M
  env STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 $extra \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*c)) MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh MP$k$c$r 600 unieinfra "$c"
  echo "$(date -u +%T) done $k c$c r$r" >> $M
}
for c in 16 4 32; do run A $c 1; run B $c 1; run B $c 2; run A $c 2; done
echo "AB MOE DONE $(date -u +%FT%TZ)" >> $M
