#!/bin/bash
# Second half of the quantization-free official-mode sweep: the concurrency
# levels test_pareto_off.sh does not cover (c1, c2, c8, c128), same 900 s
# submission-valid window and the same fresh-server-per-cell setup. Waits for
# the first half to finish before it starts, so it can be queued now.
cd /home/smc/workspace/roy
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
M=logs/pareto_off_master.log
while pgrep -f "[r]un_round.sh|test_pareto_off\.sh" >/dev/null; do sleep 10; done
echo "$(date -u +%T) OFF2 start (c1 c2 c8 c128)" >> $M
DURATION=900
for cell in 1:1 2:1 2:2 8:2 8:8 128:32 128:64 128:128; do
  conc=${cell%%:*}; cap=${cell##*:}
  echo "$(date -u +%T) start conc=$conc cap=$cap" >> $M
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh off${cap} $DURATION unieinfra "$conc"
  echo "$(date -u +%T) done conc=$conc cap=$cap rc=$?" >> $M
done
echo "PARETO-OFF2 DONE $(date -u +%FT%TZ)" >> $M
