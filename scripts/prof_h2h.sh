#!/bin/bash
# Head-to-head decode profile at the same shape (shared 130K prefix, N users), both engines at their best config.
# Waits for GPU_LOCK, holds it, writes PROFH2H DONE to logs/prof_h2h.log.
cd /home/smc/workspace/roy
LOG=logs/prof_h2h.log; LOCK=logs/GPU_LOCK; N=${N:-32}; TAG=$(date -u +%m%d%H%M)
until ( set -o noclobber; echo "owner=roy-d8(prof_h2h) since=$(date -u +%FT%TZ) release=when 'PROFH2H DONE' is in logs/prof_h2h.log" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
until ! pgrep -f "run_round.sh|serve_vllm.sh|serve_unieinfra.sh" >/dev/null; do sleep 2; done
echo "== $(date -u +%FT%TZ) lock taken N=$N tag=$TAG" >> $LOG
scripts/stop_servers.sh >/dev/null 2>&1
TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=128 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
EXTRA_SERVE_ARGS="--sampling-backend triton_full" scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $LOG.server 2>&1 && [ "$(curl -s -o /dev/null -w '%{http_code}' -X OPTIONS localhost:8000/start_profile)" != 404 ]; then
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/unie n16 /" >> $LOG
  python3 scripts/bench_decode_shared.py --n $N --max-tokens 8192 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/unie n$N /" >> $LOG &
  sleep 40
  curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d "{\"output_dir\": \"/logs/prof-h2h-u-$TAG\", \"activities\": [\"GPU\"]}" >> $LOG; echo >> $LOG
  sleep 4; curl -s -X POST localhost:8000/stop_profile >> $LOG; echo >> $LOG
  wait; sleep 20
else echo "== unieinfra server failed" >> $LOG; fi
scripts/stop_servers.sh >/dev/null 2>&1
VLLM_USE_RUST_FRONTEND=0 VLLM_PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/logs/prof-h2h-v-$TAG\",\"torch_profiler_activities\":[\"CUDA\"],\"torch_profiler_with_stack\":false,\"active_iterations\":30}" \
VLLM_MAX_NUM_SEQS=128 scripts/serve_vllm.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/vllm n16 /" >> $LOG
  python3 scripts/bench_decode_shared.py --n $N --max-tokens 8192 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/vllm n$N /" >> $LOG &
  sleep 40
  curl -s -w " start=%{http_code}" -X POST localhost:8000/start_profile >> $LOG; sleep 4; curl -s -w " stop=%{http_code}" -X POST localhost:8000/stop_profile >> $LOG; echo >> $LOG
  wait; sleep 30
else echo "== vllm server failed" >> $LOG; fi
scripts/stop_servers.sh >/dev/null 2>&1
echo "PROFH2H DONE $(date -u +%FT%TZ) tag=$TAG" >> $LOG
