#!/bin/bash
# Suffix prefill on a 100K cached prefix (AgentX turn shape) with the c32 production config (BV + prefill TBO), plus a GPU trace of one 8K suffix.
cd /home/smc/workspace/roy; LOG=logs/prof_suffix.log; LOCK=logs/GPU_LOCK; TAG=$(date -u +%m%d%H%M)
until ( set -o noclobber; echo "owner=roy-d8(prof_suffix) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK; scripts/stop_servers.sh' EXIT
scripts/stop_servers.sh
TOKENSPEED_PREFILL_TBO=${TBO:-1} TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
ENGINE_PYTHONPATH=/workspace/.wt-wave/python TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=64 \
MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 EXTRA_SERVE_ARGS="--sampling-backend triton_full" \
scripts/serve_unieinfra.sh 32 >> $LOG.server 2>&1
srv=$(ls -t logs/unieinfra-server-c32-*.log | head -1)
if scripts/wait_ready.sh 1800 "$srv" >> $LOG.server 2>&1; then
  echo "== $TAG TBO=${TBO:-1}" >> $LOG
  python3 scripts/suffix_prefill.py --sizes 1024,4096,8192 --profile-dir /logs/prof-suffix-$TAG >> $LOG 2>&1
else echo "SERVER FAILED" >> $LOG; fi
echo "SUFFIX PROF DONE $(date -u +%FT%TZ) dir=logs/prof-suffix-$TAG" >> $LOG
