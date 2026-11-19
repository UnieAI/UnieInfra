#!/bin/bash
cd /home/smc/workspace/roy
while pgrep -f "[r]un_round.sh|scripts/[t]est_k|[h]bm_server.sh" >/dev/null; do sleep 60; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
export TOKENSPEED_MIXED_REF_CHUNK=16384 TOKENSPEED_MIXED_LONG_CHUNK=16384 TOKENSPEED_MIXED_DECODE_REF=64
STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --enable-mixed-batch" scripts/run_round.sh mix94 300 unieinfra "4 16"
echo "MIX DONE $(date -u +%FT%TZ)" >> logs/round-mix94.log
