#!/bin/bash
# Level 0 (algorithm-aligned) pilot: c16, running cap 4 (the vLLM recipe value), 900 s, official mode.
#   UnieInfra: greedy draft + target-sampled exact match (triton_full, no coupling), 8192-token budget,
#              no prefill coalescing / cache-aware admission / decode-every-prefill, no INT8, util 0.9,
#              KV tokens = what vLLM chose, build pinned to the main checkout (no .wt-* kernels).
#   vLLM:      launch settings only (VLLM_ALIGN=1: greedy draft, standard rejection, 8192-token budget).
# Holds logs/GPU_LOCK for the whole run so another session does not start a server on the same GPUs/port.
cd /home/smc/workspace/roy
LOCK=logs/GPU_LOCK
M=logs/aligned_pilot.log
: > $M
# Take the lock only when no run or server is active and nobody holds it.
until [ ! -e $LOCK ] && ! pgrep -f "run_round.sh|serve_vllm.sh|serve_unieinfra.sh" >/dev/null; do sleep 2; done
echo "owner=roy-d8(aligned L0 pilot) since=$(date -u +%FT%TZ) release=when 'PILOT DONE' is written to logs/aligned_pilot.log" > $LOCK
trap 'rm -f $LOCK' EXIT
echo "$(date -u +%T) lock taken; unieinfra HEAD $(git -C unieinfra rev-parse --short HEAD)" >> $M

# 1. vLLM with the aligned launch settings: read the KV cache size it chooses (verify OUR container answers).
scripts/stop_servers.sh >> $M 2>&1
VLLM_ALIGN=1 VLLM_MAX_NUM_SEQS=4 scripts/serve_vllm.sh 16 >> $M 2>&1
scripts/wait_ready.sh 3000 >> $M 2>&1
if [ "$(docker ps -q -f name=bench-server -f status=running | wc -l)" != 1 ]; then echo "ABORT: bench-server is not running (something else answered)" >> $M; exit 1; fi
docker logs bench-server > logs/vllm-L0-server.log 2>&1
KV=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" logs/vllm-L0-server.log | head -1 | grep -oE "[0-9,]+" | tr -d ,)
grep -oE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" logs/vllm-L0-server.log | sort -u >> $M
scripts/stop_servers.sh >> $M 2>&1
[ -z "$KV" ] && { echo "ABORT: no KV size in the vLLM log" >> $M; exit 1; }
echo "$(date -u +%T) vLLM KV cache tokens: $KV" >> $M

# 2. one cell per engine, same window, official mode.
CONC=16; CAP=4; TAG=L0pilot
echo "$(date -u +%T) start unieinfra" >> $M
env -u TOKENSPEED_DSPARK_COUPLED_DRAFT -u TOKENSPEED_CACHE_AWARE_ADMISSION -u TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS \
  ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 \
  MAX_NUM_SEQS=$CAP MAX_TOTAL_TOKENS=$KV GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full" \
  scripts/run_round.sh ${TAG}u 900 unieinfra "$CONC"
echo "$(date -u +%T) done unieinfra" >> $M
echo "$(date -u +%T) start vllm" >> $M
VLLM_ALIGN=1 VLLM_MAX_NUM_SEQS=$CAP scripts/run_round.sh ${TAG}v 900 vllm "$CONC"
echo "$(date -u +%T) done vllm" >> $M
echo "PILOT DONE $(date -u +%FT%TZ)" >> $M
