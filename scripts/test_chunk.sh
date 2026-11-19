#!/bin/bash
cd /home/smc/workspace/roy
while pgrep -f "[r]un_round.sh|scripts/[t]est_k|[h]bm_server.sh" >/dev/null; do sleep 60; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
for C in 8192 4096; do
  CHUNKED_PREFILL_SIZE=$C STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh chunk$C 300 unieinfra "4 16"
done
echo "CHUNK DONE $(date -u +%FT%TZ)" >> logs/round-chunk.log
