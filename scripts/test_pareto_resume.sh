#!/bin/bash
# Resume the quantization-free official sweep after the expert-parallel probe.
# Runs the six vLLM recipe points first (c1 cap1, c4 cap2, c16 cap4, c32 cap8,
# c64 cap16, c128 cap32) because those are what the matched-interactivity
# ratio table needs, then a second cap per concurrency for interpolation.
# EP=1 selects expert parallelism; the tag then carries an "ep" suffix so the
# two configurations never get mixed up in the analysis.
cd /home/smc/workspace/roy
EP=${EP:-0}
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
if [ "$EP" = "1" ]; then export ENABLE_EXPERT_PARALLEL=1; SUF=ep; else unset ENABLE_EXPERT_PARALLEL; SUF=; fi
M=logs/pareto_off_master.log
while pgrep -f "[r]un_round.sh|test_ep_probe\.sh|test_ep_base\.sh" >/dev/null; do sleep 10; done
echo "$(date -u +%T) RESUME start EP=$EP" >> $M
for cell in 1:1 4:2 16:4 32:8 64:16 128:32 4:4 16:16 32:32 64:64 128:128 2:1 2:2 8:2 8:8; do
  conc=${cell%%:*}; cap=${cell##*:}
  echo "$(date -u +%T) start conc=$conc cap=$cap EP=$EP" >> $M
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh off${cap}${SUF} 900 unieinfra "$conc"
  echo "$(date -u +%T) done conc=$conc cap=$cap EP=$EP rc=$?" >> $M
done
echo "PARETO RESUME DONE $(date -u +%FT%TZ)" >> $M
