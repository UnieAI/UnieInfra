#!/bin/bash
# Second half of the fair sweep, in the order that closes the comparison first.
# The first half (1:1 4:2 16:4 32:8 64:16 ...) left c2/c8 with no data at all and
# c16 without a cell on the interactive side of vLLM's 128, so those come first;
# 128:32 then repeats vLLM's own cap at c128, which is the only same-cap
# comparison available at the concurrency the headline ratio comes from.
# Usage: scripts/test_fair_sweep2.sh   (run detached; logs to logs/pareto_off2.log)
cd /home/smc/workspace/roy || exit 1
L=logs/pareto_off2.log
: > "$L"
echo "FAIR SWEEP2 start $(date -u +%FT%TZ)" >> "$L"
unset ENABLE_EXPERT_PARALLEL
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1
export TOKENSPEED_CACHE_AWARE_ADMISSION=1
export TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
for cell in 2:2 8:8 16:2 128:32 64:64; do
  conc=${cell%%:*}
  cap=${cell##*:}
  echo "start conc=$conc cap=$cap $(date -u +%T)" >> "$L"
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh "off${cap}" 900 unieinfra "$conc" >> "$L" 2>&1
  echo "done conc=$conc cap=$cap rc=$? $(date -u +%T)" >> "$L"
done
echo "FAIR SWEEP2 DONE $(date -u +%FT%TZ)" >> "$L"
