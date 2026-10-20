#!/usr/bin/env bash
# E48 driver-of-drivers: run the remaining phases back-to-back on ONE held GPU.
#
# Each phase is a separate `run_e48_fp8_kv.sh` invocation (they are long, and a
# phase boundary is the only safe place to change the arm list or `--max-model-len`).
# `KEEP_HOLD=1` on all but the last keeps `e48_gpu_hold.py` resident across the
# boundary, so the card is never released to the five other agents on this box.
#
# This exists as a FILE rather than an inline `bash -c` because the first attempt
# at chaining was written inline and died on nested quoting, silently, leaving the
# GPU held and idle for ten minutes.
set -uo pipefail

REPO=/workspace/roy/UnieInfra
cd "$REPO"
LOGS=$REPO/benchmark/results/_srvlogs/e48
GPU=${GPU:-1}
ARMS_ALL=${ARMS_ALL:-"bf16fa bf16tri bf16fi e4m3tri e4m3auto"}

# bf16fi was added after the base phase had already run the other arms: the base
# phase is what revealed that fp8 does not land on Triton by default at all, but
# on FlashInfer, so a bf16/FlashInfer control became necessary.
GPU=$GPU KEEP_HOLD=1 PHASES="base" ARMS="bf16fi" \
  bash benchmark/scripts/run_e48_fp8_kv.sh > "$LOGS/E48_BASE_FI.log" 2>&1

GPU=$GPU KEEP_HOLD=1 PHASES="tput" REPS=${REPS:-3} ARMS="$ARMS_ALL" \
  bash benchmark/scripts/run_e48_fp8_kv.sh > "$LOGS/E48_TPUT.log" 2>&1

GPU=$GPU PHASES="long" LONG_REPS=${LONG_REPS:-3} ARMS="$ARMS_ALL" \
  bash benchmark/scripts/run_e48_fp8_kv.sh > "$LOGS/E48_LONG.log" 2>&1

echo "=== E48 all phases complete $(date -u +%FT%TZ) ==="
