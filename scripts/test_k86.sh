#!/bin/bash
cd /home/smc/workspace/roy
while pgrep -f "test_k8[45].sh" >/dev/null || pgrep -f "run_round.sh (k84|diag85|stressbig)" >/dev/null; do sleep 10; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
MAX_TOTAL_TOKENS=41943040 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh k86 300 unieinfra "128 64"
echo "K86 DONE $(date -u +%FT%TZ)" >> logs/round-k86.log
