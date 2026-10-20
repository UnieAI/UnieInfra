#!/usr/bin/env bash
# E36: rotating back-to-back N=3 comparison of four scheduling arms on GPU 3.
#
# Rotation (A,B,C,D) x 3 rather than 3xA then 3xB, so any slow drift in box
# conditions is spread across arms instead of being confounded with one.
#
#   A stock    - stock vLLM scheduler (no --scheduler-cls at all)
#   B thr      - UnieInfra policy, throughput objective (prefill gate on, hold 16)
#   C inter    - UnieInfra policy, interactive objective (gate off, cap only)
#   D static   - UnieInfra policy, static-cadence ablation (no load/SLO signals)
#
# Every vLLM flag is identical across arms, so all four share one torch.compile
# cache entry and replay byte-identical graphs (E34's cache key is `VllmConfig`,
# which `scheduler_cls` does not enter). Only the policy differs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CLS=unieinfra_scheduler.vllm_scheduler.AdaptiveScheduler
PREFIX="${PREFIX:-e36}"

run() {  # run <label> <objective|stock> [extra env assignments]
  local label="$1" obj="$2"; shift 2
  if [ "$obj" = "stock" ]; then
    LABEL="$label" bash "$REPO/benchmark/scripts/sched_arm.sh"
  else
    env "$@" LABEL="$label" SCHED_CLS="$CLS" SCHED_OBJECTIVE="$obj" \
      bash "$REPO/benchmark/scripts/sched_arm.sh"
  fi
}

for R in 1 2 3; do
  run "${PREFIX}_stock_r${R}"  stock
  run "${PREFIX}_thr_r${R}"    throughput  UNIEINFRA_SCHED_LOG_EVERY=1000
  run "${PREFIX}_inter_r${R}"  interactive
  run "${PREFIX}_static_r${R}" static      UNIEINFRA_SCHED_HOLD_STEPS=4
done
echo "[e36] all arms complete"
