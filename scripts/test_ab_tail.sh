#!/bin/bash
cd /home/smc/workspace/roy
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
for R in 2 32; do
  TOKENSPEED_V41_TAIL_ROWS=$R STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh abt$R 300 unieinfra "16 64"
done
echo "ABT DONE $(date -u +%FT%TZ)" >> logs/round-abt.log
