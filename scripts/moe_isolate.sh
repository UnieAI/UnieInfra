#!/bin/bash
# c$1 GPU_STEP_TIMING (no profiler): mi350p-final with prefill MoE v2 off vs on, OFF ON ON OFF.
cd /home/smc/workspace/roy; C=${1:-4}
OFF="TOKENSPEED_MOE_PREFILL_S1_V2=0 TOKENSPEED_MOE_PREFILL_S2_V2=0 TOKENSPEED_MOE_PREFILL_S2_REDUCE_V2=0"
for L in OFF1 ON1 ON2 OFF2; do
  if [[ $L == OFF* ]]; then E=$OFF; else E="TOKENSPEED_MOE_PREFILL_S1_V2=1"; fi
  NOPROF=1 scripts/prof_final.sh $C ${L}c$C TOKENSPEED_GPU_STEP_TIMING=1 $E
done
echo "ISOLATE DONE c$C $(date -u +%FT%TZ)" >> logs/prof_final.log
