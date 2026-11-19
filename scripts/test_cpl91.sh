#!/bin/bash
# Draft temperature scale sweep for coupled drafting at c16 (lossless knob; moves acceptance).
cd /home/smc/workspace/roy
while pgrep -f "run_round.sh" >/dev/null; do sleep 1; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
for S in 0.7 1.4; do
  TOKENSPEED_DSPARK_COUPLED_TEMP_SCALE=$S MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" scripts/run_round.sh cpl91s${S/./p} 300 unieinfra "16"
done
echo "CPL91 DONE $(date -u +%FT%TZ)" >> logs/round-cpl91.log
