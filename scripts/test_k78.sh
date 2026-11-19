#!/bin/bash
cd /home/smc/workspace/roy
while pgrep -f "run_round.sh k77b" >/dev/null; do sleep 10; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
GPU_MEMORY_UTILIZATION=0.97 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh k78 300 unieinfra "128 64 16 1"
echo "K78 DONE $(date -u +%FT%TZ)" >> logs/round-k78.log
