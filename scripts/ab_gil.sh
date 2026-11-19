#!/bin/bash
# A/B TOKENSPEED_GIL_SWITCH_INTERVAL_US (73d2cfdb, worktree .wt-gil): default 5000 us vs 500 vs 200, order 0/500/200/0/500/200.
# Trace finding (prof-seg1-11121241): zero_cache_pages' Event.wait stalls 2.25 ms per decode step exactly while the
# scheduler thread runs post_process; the GPU idles 2.7 ms there.
cd /home/smc/workspace/roy; LOG=logs/ab_gil.log; LOCK=logs/GPU_LOCK
until grep -q "BVQ DONE" logs/bv_quality.log 2>/dev/null; do sleep 20; done
until ( set -o noclobber; echo "owner=roy-d8(ab_gil) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
for G in 0 500 200 0 500 200; do
  scripts/stop_servers.sh >/dev/null 2>&1
  GE=""; [ $G != 0 ] && GE="TOKENSPEED_GIL_SWITCH_INTERVAL_US=$G"
  env $GE HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-gil/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=128 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full" scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $LOG.server 2>&1 || { echo "G=$G server failed" >> $LOG; continue; }
  echo "G=$G $(grep -m1 -o 'GIL switch interval set to [0-9]* us' logs/unieinfra-server.log)" >> $LOG
  for n in 1 16 32; do
    python3 scripts/bench_decode_shared.py --n $n --max-tokens 1024 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/G=$G /" >> $LOG
  done
done
scripts/stop_servers.sh >/dev/null 2>&1
echo "AB DONE $(date -u +%FT%TZ)" >> $LOG
