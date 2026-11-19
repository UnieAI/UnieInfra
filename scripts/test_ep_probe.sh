#!/bin/bash
# Expert-parallel probe. The unquantized cells so far run MoE with
# tensor-parallel sharding (ep_size=1, --enable-expert-parallel absent), which
# makes every FFN pay a TP all-reduce of the MoE output. Expert parallelism
# keeps whole experts per rank and swaps that all-reduce for a small
# all-to-all, and it is numerically exact (no quantization), so it is legal
# under the fair-comparison rule. Two 900 s cells at the concurrency where the
# MoE reduce is amortized least, enough to see whether the lever is worth a
# full sweep before spending hours on one shape or the other.
cd /home/smc/workspace/roy
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0 ENABLE_EXPERT_PARALLEL=1
M=logs/pareto_off_master.log
while pgrep -f "[r]un_round.sh|test_pareto_off\.sh" >/dev/null; do sleep 10; done
echo "$(date -u +%T) EP PROBE start (expert parallel on)" >> $M
for cell in 32:32 128:128; do
  conc=${cell%%:*}; cap=${cell##*:}
  echo "$(date -u +%T) start conc=$conc cap=$cap ep=1" >> $M
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh off${cap}ep 900 unieinfra "$conc"
  echo "$(date -u +%T) done conc=$conc cap=$cap ep=1 rc=$?" >> $M
done
echo "EP PROBE DONE $(date -u +%FT%TZ)" >> $M
