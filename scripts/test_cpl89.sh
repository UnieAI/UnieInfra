#!/bin/bash
# Coupled drafting: the DSpark block is drawn with the verifier's own Gumbel noise
# (TOKENSPEED_DSPARK_COUPLED_DRAFT=1) under --sampling-backend triton_full.
# Same build and cache as samp88; compare accept length and RPM with samp88 (2.83 / 41.2 at c4).
cd /home/smc/workspace/roy
while pgrep -f "run_round.sh" >/dev/null; do sleep 15; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full --enable-log-request-stats" scripts/run_round.sh cpl89 300 unieinfra "4"
echo "CPL89 DONE $(date -u +%FT%TZ)" >> logs/round-cpl89.log
