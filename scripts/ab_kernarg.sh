#!/bin/bash
# A/B HIP_FORCE_DEV_KERNARG on the same decode shape (shared 130K prefix, T=1.0), best UnieInfra config.
cd /home/smc/workspace/roy; LOG=logs/ab_kernarg.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(ab_kernarg) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
for KA in 1 0 1 0; do
  scripts/stop_servers.sh >/dev/null 2>&1
  HIP_FORCE_DEV_KERNARG=$KA TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=128 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full" scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $LOG.server 2>&1 || { echo "KA=$KA server failed" >> $LOG; continue; }
  docker exec ts-dev true 2>/dev/null
  for n in 1 16 32; do
    python3 scripts/bench_decode_shared.py --n $n --max-tokens 1024 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/KA=$KA /" >> $LOG
  done
done
scripts/stop_servers.sh >/dev/null 2>&1
echo "AB DONE $(date -u +%FT%TZ)" >> $LOG
