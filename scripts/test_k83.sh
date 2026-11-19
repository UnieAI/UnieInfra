#!/bin/bash
cd /home/smc/workspace/roy
while pgrep -f "run_round.sh k82" >/dev/null || pgrep -f "test_k82.sh" >/dev/null; do sleep 10; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
MAX_TOTAL_TOKENS=47185920 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh k83 300 unieinfra "128 64"
echo "K83 DONE $(date -u +%FT%TZ)" >> logs/round-k83.log
