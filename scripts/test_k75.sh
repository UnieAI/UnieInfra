#!/bin/bash
cd /home/smc/workspace/roy
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh k75 300 unieinfra "1 4 16 32 64 128"
echo "K75 DONE $(date -u +%FT%TZ)" >> logs/round-k75.log
