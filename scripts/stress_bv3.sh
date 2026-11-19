#!/bin/bash
# AgentX stress (STRESS=1, 600 s) UnieInfra with block verification (.wt-bv e966e7a5, TOKENSPEED_DSPARK_BLOCK_VERIFY=1),
# same config as the stress best-vs-best UnieInfra cells (cap 2x CONC, KV uncapped). Compare with the SB* vLLM cells.
cd /home/smc/workspace/roy; M=logs/stress_bv.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress_bv) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
for c in 128; do
  echo "$(date -u +%T) start unieinfra BV c$c" >> $M
  STRESS=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-bv/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*c)) MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh SBV${c}k33 600 unieinfra "$c"
  echo "$(date -u +%T) done unieinfra BV c$c" >> $M
done
echo "STRESS BV3 DONE $(date -u +%FT%TZ)" >> $M
