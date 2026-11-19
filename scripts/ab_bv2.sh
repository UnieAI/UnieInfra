#!/bin/bash
# (1) Block-verify A/B with the TP-group fix (e966e7a5, worktree .wt-bv): F=1, F=0, F=1.
# (2) UnieInfra decode GPU traces with DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 vs 2 (n=32, same shape as prof_h2h),
#     to see whether forcing packet-batch graph submission removes the GPU idle between graphs.
cd /home/smc/workspace/roy; LOG=logs/ab_bv2.log; LOCK=logs/GPU_LOCK; TAG=$(date -u +%m%d%H%M)
until ( set -o noclobber; echo "owner=roy-d8(ab_bv2+prof_seg) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
BASE_ENV="HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=128 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9"
start() {  # name extra-env...
  local name=$1; shift
  scripts/stop_servers.sh >/dev/null 2>&1
  env $BASE_ENV "$@" EXTRA_SERVE_ARGS="--sampling-backend triton_full" scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $LOG.server 2>&1 && return 0
  echo "$name server failed: $(grep -m2 -E 'Error:|KeyError|RuntimeError' logs/unieinfra-server.log | cut -c1-220)" >> $LOG; return 1
}
BV=ENGINE_PYTHONPATH=/workspace/.wt-bv/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor
for F in 1 0 1; do
  start "BV F=$F" $BV TOKENSPEED_DSPARK_BLOCK_VERIFY=$F || continue
  for n in 1 16 32; do
    python3 scripts/bench_decode_shared.py --n $n --max-tokens 1024 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/F=$F /" >> $LOG
    echo "F=$F n=$n $(grep -oE 'accept_len: *[0-9.]+' logs/unieinfra-server.log | tail -5 | tr '\n' ' ')" >> $LOG
  done
done
for SEG in 1 2; do
  start "PROF SEG=$SEG" ENGINE_PYTHONPATH=/nonexistent DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=$SEG || continue
  python3 scripts/bench_decode_shared.py --n 32 --max-tokens 8192 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/PROF SEG=$SEG /" >> $LOG &
  sleep 40
  curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d "{\"output_dir\": \"/logs/prof-seg$SEG-$TAG\", \"activities\": [\"GPU\", \"CPU\"]}" >> $LOG; echo >> $LOG
  sleep 4; curl -s -X POST localhost:8000/stop_profile >> $LOG; echo >> $LOG
  wait; sleep 20
done
scripts/stop_servers.sh >/dev/null 2>&1
echo "AB DONE $(date -u +%FT%TZ) tag=$TAG" >> $LOG
