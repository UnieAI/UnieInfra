#!/bin/bash
# Best config so far, missing cells: coupled DSpark drafting + 4K prefill chunks + INT8 prefill all-reduce
# (serve_unieinfra.sh defaults) with T=1.0 sampling. c16-c64 come from cpl90 (same settings).
cd /home/smc/workspace/roy
while pgrep -f "run_round.sh" >/dev/null; do sleep 1; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" scripts/run_round.sh best1 300 unieinfra "1 4"
MAX_TOTAL_TOKENS=41943040 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" scripts/run_round.sh best1 300 unieinfra "128"
echo "BEST1 DONE $(date -u +%FT%TZ)" >> logs/round-best1.log
