#!/bin/bash
# Baseline for cpl90: independent sampling (no coupling) at the same 4K prefill chunks, c16-c64.
cd /home/smc/workspace/roy
while pgrep -f "run_round.sh" >/dev/null; do sleep 1; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
CHUNKED_PREFILL_SIZE=4096 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" scripts/run_round.sh samp91 300 unieinfra "16 32 64"
echo "SAMP91 DONE $(date -u +%FT%TZ)" >> logs/round-samp91.log
