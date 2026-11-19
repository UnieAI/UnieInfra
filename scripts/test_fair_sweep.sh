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
# Every cell here is tensor-parallel MoE: the fair c128 baseline (id 120) showed
# expert parallelism is ~25% worse in RPM even at cap 128, so EP is dropped.
cd /home/smc/workspace/roy
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
M=logs/pareto_off_master.log
while pgrep -f "[r]un_round.sh|test_ep_probe\.sh|test_ep_base\.sh|test_pareto_resume\.sh" >/dev/null; do sleep 10; done
echo "$(date -u +%T) FAIR SWEEP start" >> $M
for cell in 1:1 4:2 16:4 32:8 64:16 128:32 4:4 16:16 32:32 2:2 8:8 64:64; do
  conc=${cell%%:*}; cap=${cell##*:}; unset ENABLE_EXPERT_PARALLEL
  echo "$(date -u +%T) start conc=$conc cap=$cap" >> $M
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh off${cap} 900 unieinfra "$conc"
  echo "$(date -u +%T) done conc=$conc cap=$cap rc=$?" >> $M
done
echo "FAIR SWEEP DONE $(date -u +%FT%TZ)" >> $M
