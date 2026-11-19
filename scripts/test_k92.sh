#!/bin/bash
cd /home/smc/workspace/roy
while pgrep -f "[r]un_round.sh|scripts/[t]est_k79|[h]bm_server.sh" >/dev/null; do sleep 60; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=1
MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh k92 300 unieinfra "1 4 16 32 64 128"
echo "K92 DONE $(date -u +%FT%TZ)" >> logs/round-k92.log
