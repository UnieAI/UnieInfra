#!/bin/bash
# Official-mode throughput vs interactivity sweep, submission-valid duration.
# Recorded inter-turn gaps (no --ignore-trace-delays), 900 s per cell, fresh
# server per cell. Caps bracket the vLLM recipe point (CONC/4) and extend up to
# CONC, which is where matched-interactivity comparison happens.
# Quantization is off on purpose: the INT8 all-reduce path (#93) is a lossy
# comm trick that vLLM does not use, so it may not appear in a fair comparison.
cd /home/smc/workspace/roy
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
M=logs/pareto_off_master.log
# Wait for any in-flight round without matching this script's own argv.
while pgrep -f "[r]un_round.sh|scripts/[t]est_[a-z]*\.sh" >/dev/null; do sleep 5; done
DURATION=900
for conc in 4 16 32 64; do
  q4=$((conc/4)); [ "$q4" -lt 1 ] && q4=1
  for cap in $q4 $((conc/2)) $conc; do
    echo "$(date -u +%T) start conc=$conc cap=$cap" >> $M
    MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
      EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
      scripts/run_round.sh off${cap} $DURATION unieinfra "$conc"
    echo "$(date -u +%T) done conc=$conc cap=$cap rc=$?" >> $M
  done
done
echo "PARETO-OFF DONE $(date -u +%FT%TZ)" >> $M
