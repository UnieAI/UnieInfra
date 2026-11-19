#!/bin/bash
# A/B TOKENSPEED_HOLD_RESULTS_UNTIL_LAUNCH (.wt-hostfix): one window of flags "$@" (e.g. 1 0),
# n=32 decode bench each; PROF=1 also records a 4 s CPU+GPU trace on the first flag=1 server.
# Unit tests (need a visible GPU for platform detection) run first when TESTS=1.
cd /home/smc/workspace/roy; LOG=logs/ab_hold.log; LOCK=logs/GPU_LOCK; TAG=$(date -u +%m%d%H%M)
until ( set -o noclobber; echo "owner=hostfix-agent(ab_hold $*) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'scripts/stop_servers.sh >/dev/null 2>&1; rm -f $LOCK' EXIT
echo "WINDOW $* start $(date -u +%FT%TZ)" >> $LOG
if [ "${TESTS:-0}" = 1 ]; then
  docker exec -w /workspace/.wt-hostfix ts-dev bash -c "PYTHONPATH=/workspace/.wt-hostfix/python timeout 600 python3 -m pytest -q -p no:cacheprovider test/runtime/test_forward_thread.py test/runtime/test_device_handle.py test/runtime/test_forward_routing.py 2>&1 | tail -15" >> $LOG 2>&1
fi
BASE_ENV="HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=128 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 ENGINE_PYTHONPATH=/workspace/.wt-hostfix/python"
prof_done=0
for F in "$@"; do
  scripts/stop_servers.sh >/dev/null 2>&1
  env $BASE_ENV TOKENSPEED_HOLD_RESULTS_UNTIL_LAUNCH=$F EXTRA_SERVE_ARGS="--sampling-backend triton_full" scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  if ! scripts/wait_ready.sh 1200 logs/unieinfra-server.log >> $LOG.server 2>&1; then
    echo "H=$F server failed: $(grep -m2 -E 'Error:|KeyError|RuntimeError' logs/unieinfra-server.log | cut -c1-220)" >> $LOG; continue
  fi
  python3 scripts/bench_decode_shared.py --n 32 --max-tokens 1024 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/H=$F /" >> $LOG
  echo "H=$F $(grep -oE 'accept_len: *[0-9.]+' logs/unieinfra-server.log | tail -3 | tr '\n' ' ')" >> $LOG
  if [ "${PROF:-0}" = 1 ]; then
    prof_done=1
    python3 scripts/bench_decode_shared.py --n 32 --max-tokens 4096 --seed 8 --temperature 1.0 2>&1 | tail -1 | sed "s/^/PROF H=$F /" >> $LOG &
    sleep 25
    curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d "{\"output_dir\": \"/logs/prof-hold$F-$TAG\", \"activities\": [\"GPU\", \"CPU\"]}" >> $LOG; echo >> $LOG
    sleep 4; curl -s -X POST localhost:8000/stop_profile >> $LOG; echo >> $LOG
    wait
  fi
done
echo "WINDOW $* done $(date -u +%FT%TZ) tag=$TAG" >> $LOG
