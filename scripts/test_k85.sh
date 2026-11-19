#!/bin/bash
# Diagnostic run (not a performance data point): per-request cache hit stats and shadow n-gram acceptance.
cd /home/smc/workspace/roy
while pgrep -f "test_k8[234].sh" >/dev/null || pgrep -f "run_round.sh k8[234]" >/dev/null || pgrep -f "run_round.sh stressbig" >/dev/null; do sleep 10; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_SHADOW_NGRAM=1
MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --enable-log-request-stats" scripts/run_round.sh diag85 300 unieinfra "4 32"
echo "K85 DONE $(date -u +%FT%TZ)" >> logs/round-diag85.log
