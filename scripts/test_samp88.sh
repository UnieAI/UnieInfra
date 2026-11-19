#!/bin/bash
# Same build and cache size as k79, but honouring per-request temperature (AgentX sends 1.0):
# the AMD default 'greedy' backend ignores it, which makes speculative acceptance easier.
cd /home/smc/workspace/roy
while pgrep -f "test_k8[0-9].sh" >/dev/null || pgrep -f "test_prefix_probe.sh" >/dev/null || pgrep -f "run_round.sh" >/dev/null; do sleep 10; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" scripts/run_round.sh samp88 300 unieinfra "4 16 32 64"
echo "SAMP88 DONE $(date -u +%FT%TZ)" >> logs/round-samp88.log
