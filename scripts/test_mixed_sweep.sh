#!/bin/bash
# Fair (no-quantization) official sweep with a per-cell MoE layout.
#
# The EP probe showed the expert-parallel layout is not universally better:
#   c32 cap32 -> TP-MoE 69.8 RPM @ interactivity 43 vs EP 50.9 @ 27 (TP wins)
#   c128 cap128 -> EP 96.4 @ 10 (TP cell still running, expected far lower)
# so the sweep picks the layout per cell: expert parallelism only where the
# decode batch is large (cap >= 64), tensor-parallel MoE everywhere else.
# Both layouts are exact arithmetic, so the comparison stays quantization-free.
#
# Cell spec: conc:cap:ep   (ep=1 -> tag suffix "ep", ENABLE_EXPERT_PARALLEL=1)
cd /home/smc/workspace/roy
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
M=logs/pareto_off_master.log
while pgrep -f "[r]un_round.sh|test_ep_probe\.sh|test_ep_base\.sh|test_pareto_resume\.sh" >/dev/null; do sleep 10; done
echo "$(date -u +%T) MIXED SWEEP start" >> $M
for cell in 1:1:0 4:2:0 16:4:0 32:8:0 64:16:0 128:32:0 4:4:0 16:16:0 32:32:0 2:2:0 8:8:0 64:64:1 128:128:1; do
  conc=${cell%%:*}; rest=${cell#*:}; cap=${rest%%:*}; ep=${rest##*:}
  if [ "$ep" = "1" ]; then export ENABLE_EXPERT_PARALLEL=1; SUF=ep; else unset ENABLE_EXPERT_PARALLEL; SUF=; fi
  echo "$(date -u +%T) start conc=$conc cap=$cap ep=$ep" >> $M
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh off${cap}${SUF} 900 unieinfra "$conc"
  echo "$(date -u +%T) done conc=$conc cap=$cap ep=$ep rc=$?" >> $M
done
echo "MIXED SWEEP DONE $(date -u +%FT%TZ)" >> $M
