#!/bin/bash
# A/B TOKENSPEED_DECODE_HOST_PREP (branch decode-host-gap 537d92ef, worktree .wt-hgap), same code both arms, order 1/0/1/0.
cd /home/smc/workspace/roy; LOG=logs/ab_hostprep.log; LOCK=logs/GPU_LOCK
until grep -q "STRESS DONE" logs/stress_bvb.log 2>/dev/null; do sleep 30; done
until ( set -o noclobber; echo "owner=roy-d8(ab_hostprep) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
for F in 1 0 1 0; do
  scripts/stop_servers.sh >/dev/null 2>&1
  TOKENSPEED_DECODE_HOST_PREP=$F HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-hgap/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=128 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full" scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $LOG.server 2>&1 || { echo "F=$F server failed" >> $LOG; continue; }
  for n in 1 16 32; do
    python3 scripts/bench_decode_shared.py --n $n --max-tokens 1024 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/F=$F /" >> $LOG
  done
done
scripts/stop_servers.sh >/dev/null 2>&1
echo "AB DONE $(date -u +%FT%TZ)" >> $LOG
