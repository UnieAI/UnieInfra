#!/bin/bash
cd /home/smc/workspace/roy
# Serialize with any other sweep, but not with ourselves: "scripts/[t]est_k[0-9]"
# matches this script's own command line, so a bare pgrep never falls through
# (the watcher started on 2026-11-08 sat here for 31 h and k100 never ran).
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" scripts/run_round.sh k100 300 unieinfra "1 4 16 32 64 128"
echo "K100 DONE $(date -u +%FT%TZ)" >> logs/round-k100.log
