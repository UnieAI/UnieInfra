#!/bin/bash
# AgentX stress 600 s: block verify + aiter MoE with the fragmentation fix (moe-aiter-fix bbb32bee, worktree .wt-aitfix).
cd /home/smc/workspace/roy; M=logs/stress_bv.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress_bvf) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
for c in 4 16 32 64 128 1; do
  echo "$(date -u +%T) start unieinfra BVaiterfix2 c$c" >> $M
  STRESS=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 MOE_BACKEND=aiter \
  ENGINE_PYTHONPATH=/workspace/.wt-aitfix/python:/workspace/.wt-aitfix/tokenspeed-kernel/python:/workspace/.vendor TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*c)) MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh SBF$c 600 unieinfra "$c"
  f=$(ls -t logs/unieinfra-server-c$c-*.log | head -1)
  echo "$(date -u +%T) done unieinfra BVaiterfix2 c$c $(grep -m1 -oE 'token_capacity=[0-9]+' $f) $(grep -m1 -oE 'moe_backend=[a-z]+' $f)" >> $M
done
echo "STRESS BVF DONE $(date -u +%FT%TZ)" >> $M
