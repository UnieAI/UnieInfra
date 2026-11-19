#!/usr/bin/env bash
# Reproducibility A/B for the c64 cap16 cell: same traffic window, two MoE
# backends back to back. run_round.sh starts a fresh server per cell.
set -x
cd /home/smc/workspace/roy
export MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=0.90
TOKENSPEED_KERNEL_VERBOSE=1 bash scripts/run_round.sh repait-c64 900 "unieinfra" "64"
TOKENSPEED_KERNEL_VERBOSE=1 TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY=triton \
  bash scripts/run_round.sh reptr64 900 "unieinfra" "64"
echo REP_AB_DONE
