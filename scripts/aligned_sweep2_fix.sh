#!/bin/bash
# Re-run the UnieInfra c1 cell of the fair sweep (the first attempt was killed by my own stop_servers.sh at 16:16).
# Same settings as scripts/aligned_sweep2.sh; waits for 'SWEEP DONE', then takes the GPU lock.
cd /home/smc/workspace/roy
LOCK=logs/GPU_LOCK
M=logs/aligned_sweep2_fix.log
: > $M
until grep -q "SWEEP DONE" logs/aligned_sweep2.log 2>/dev/null; do sleep 20; done
until ( set -o noclobber; echo "owner=roy-d8(fair sweep: unieinfra c1 rerun) since=$(date -u +%FT%TZ) release=when 'FIX DONE' is written to logs/aligned_sweep2_fix.log" > $LOCK ) 2>/dev/null; do sleep 5; done
trap 'rm -f $LOCK' EXIT
until ! pgrep -f "run_round.sh|serve_vllm.sh|serve_unieinfra.sh" >/dev/null; do sleep 2; done
echo "$(date -u +%T) start unieinfra c1 (cap 1, budget 16384)" >> $M
TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=16384 \
MAX_NUM_SEQS=1 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-mixed-batch" \
scripts/run_round.sh FS1u2 900 unieinfra 1
echo "$(date -u +%T) done unieinfra c1" >> $M
echo "FIX DONE $(date -u +%FT%TZ)" >> $M
