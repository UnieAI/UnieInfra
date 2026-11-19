#!/bin/bash
# Two things that have to happen after the vLLM reference rerun, in one detached
# queue so nothing idles between them:
#
# 1. Cap ladder (16:2, 32:4, 64:8). Every current row is measured "near": our
#    nearest cell sits at LOWER interactivity than the vLLM reference (c16 117 vs
#    128, c32 78 vs 82, c64 49 vs 55), so the ratio is read off a cell that is
#    slower per user than the thing it is compared against - generous to us. One
#    smaller cap per concurrency brackets the reference interactivity from above
#    and lets fair.py interpolate instead of taking the nearest point. c1 and c4
#    already bracket (c4 146 vs 145); c128 cannot (cap 128 is the ceiling).
#
# 2. Mixed prefill/decode cells (scripts/test_mixed_official.sh, c32 cap8 and
#    c64 cap16), the one untested scheduling lever at the two cells we lose.
cd /home/smc/workspace/roy || exit 1
L=logs/after_vllm.log
: > "$L"
echo "AFTER-VLLM QUEUE start $(date -u +%FT%TZ), waiting for the vLLM reference rerun" >> "$L"
while pgrep -f "[t]est_vllm_ref.sh" >/dev/null; do sleep 30; done
echo "vLLM reference rerun finished $(date -u +%FT%TZ)" >> "$L"

export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
for cell in 16:2 32:4 64:8; do
  conc=${cell%%:*}; cap=${cell##*:}
  echo "$(date -u +%T) ladder start conc=$conc cap=$cap" >> "$L"
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh off${cap} 900 unieinfra "$conc"
  echo "$(date -u +%T) ladder done conc=$conc cap=$cap rc=$?" >> "$L"
done
echo "LADDER DONE $(date -u +%FT%TZ)" >> "$L"
scripts/test_mixed_official.sh
echo "AFTER-VLLM QUEUE DONE $(date -u +%FT%TZ)" >> "$L"
