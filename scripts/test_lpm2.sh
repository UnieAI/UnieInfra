#!/bin/bash
# c128 with cache-aware admission and fewer resident requests (64 / 48).
cd /home/smc/workspace/roy
until grep -q "OVERLAP DONE" logs/overlap-iris-ab.log 2>/dev/null; do sleep 30; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
for s in 64 48; do
  STRESS=1 MAX_NUM_SEQS=$s EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh "lpmseq$s" 300 unieinfra 128
done
echo "LPM2 DONE $(date -u +%FT%TZ)" >> logs/round-lpm.log
