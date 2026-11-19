#!/bin/bash
# Throughput vs interactivity sweep in the official mode (recorded inter-turn gaps, no --ignore-trace-delays):
# for each concurrency, three running-request caps (CONC/4, CONC/2, 2xCONC), 600 s each.
# Best configuration so far: coupled DSpark drafting + T=1 sampling, 4K chunks, INT8 prefill all-reduce.
cd /home/smc/workspace/roy
while pgrep -f "run_round.sh" >/dev/null; do sleep 1; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
M=logs/pareto_master.log
for conc in 4 16 32 64; do
  for kind in q4 h2 x2; do
    case $kind in q4) cap=$((conc/4));; h2) cap=$((conc/2));; x2) cap=$((conc*2));; esac
    echo "$(date -u +%T) start conc=$conc cap=$cap" >> $M
    MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" scripts/run_round.sh parS${cap} 600 unieinfra "$conc"
    echo "$(date -u +%T) done conc=$conc cap=$cap" >> $M
  done
done
echo "PARETO DONE $(date -u +%FT%TZ)" >> $M
