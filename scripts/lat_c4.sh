#!/bin/bash
# Low-load latency breakdown: the fair-sweep UnieInfra c4 cell again (same settings and 900 s window as FS4u), with
# per-request stats enabled (queue / prefill / TTFT per request). Holds GPU_LOCK.
cd /home/smc/workspace/roy
LOCK=logs/GPU_LOCK
M=logs/lat_c4.log
: > $M
until ( set -o noclobber; echo "owner=roy-d8(c4 latency breakdown) since=$(date -u +%FT%TZ) release=when 'LAT DONE' is written to logs/lat_c4.log" > $LOCK ) 2>/dev/null; do sleep 3; done
trap 'rm -f $LOCK' EXIT
until ! pgrep -f "run_round.sh|serve_vllm.sh|serve_unieinfra.sh" >/dev/null; do sleep 2; done
echo "$(date -u +%T) start unieinfra c4 with request stats" >> $M
TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=16384 \
MAX_NUM_SEQS=8 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-mixed-batch --enable-log-request-stats" \
scripts/run_round.sh LAT4u 900 unieinfra 4
echo "$(date -u +%T) done" >> $M
echo "LAT DONE $(date -u +%FT%TZ)" >> $M
