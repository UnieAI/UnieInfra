#!/bin/bash
# Prefill eager-gap A/B at c32 (AgentX stress 600 s, PR build + block verify), 2 runs each, interleaved:
#   P8  = chunk 8192, prefill graph max 2048 (current: 8K chunks run eager)
#   P4  = chunk 4096 (earlier best chunk)
#   P8G = chunk 8192 + --prefill-graph-max-tokens 8192
cd /home/smc/workspace/roy; M=logs/prefill_ab.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(prefill_ab) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() {  # tag chunk extra-args c rep
  echo "$(date -u +%T) start $1 c$4 r$5" >> $M
  STRESS=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-clean/python TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=$2 MAX_NUM_SEQS=$((2*$4)) \
  MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats $3" \
  scripts/run_round.sh PA$1$4$5 600 unieinfra "$4"
  echo "$(date -u +%T) done $1 c$4 r$5 $(grep -m1 -oE 'captured decoder buckets \[[^]]*\]' $(ls -t logs/unieinfra-server-c$4-*.log | head -1))" >> $M
}
for r in 1 2; do
  run P8 8192 "" 32 $r
  run P4 4096 "" 32 $r
  run P8G 8192 "--prefill-graph-max-tokens 8192" 32 $r
done
echo "PREFILL AB DONE $(date -u +%FT%TZ)" >> $M
