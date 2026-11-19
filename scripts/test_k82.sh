#!/bin/bash
cd /home/smc/workspace/roy
while pgrep -f "run_round.sh k79|run_round.sh k81" >/dev/null || pgrep -f "test_k81.sh" >/dev/null; do sleep 10; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_VERIFY_DRAFTS=3
MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh k82 300 unieinfra "128 64 32 16 4"
echo "K82 DONE $(date -u +%FT%TZ)" >> logs/round-k82.log
