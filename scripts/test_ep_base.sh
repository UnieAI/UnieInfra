#!/bin/bash
# Baseline cells for the expert-parallel probe: the same shapes the probe runs
# (c32 cap32, c128 cap128) with the tensor-parallel MoE layout that every other
# fair cell uses. Without these two cells the probe result has no same-shape
# reference, and both of them are needed for the main curve anyway.
cd /home/smc/workspace/roy
unset ENABLE_EXPERT_PARALLEL
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
M=logs/pareto_off_master.log
while pgrep -f "[r]un_round.sh|test_ep_probe\.sh" >/dev/null; do sleep 10; done
echo "$(date -u +%T) EP BASELINE start (tensor-parallel MoE, EP off)" >> $M
for cell in 32:32 128:128; do
  conc=${cell%%:*}; cap=${cell##*:}
  echo "$(date -u +%T) start conc=$conc cap=$cap EP=0" >> $M
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh off${cap} 900 unieinfra "$conc"
  echo "$(date -u +%T) done conc=$conc cap=$cap EP=0 rc=$?" >> $M
done
echo "EP BASELINE DONE $(date -u +%FT%TZ)" >> $M
