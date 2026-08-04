#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
#
# Hardware-counter comparison of the three decode-GEMV contenders (E33),
# capturing the same metrics E26 used so the numbers are directly comparable.
#
# Counters are locked by the driver's RmProfilingAdminOnly plus a missing
# CAP_SYS_ADMIN in the normal container, so this must run inside a *privileged*
# exec session. That is non-destructive: it does not restart the container and
# does not touch other GPUs or other users' jobs.
#
#   docker exec --privileged -e CUDA_VISIBLE_DEVICES=<free gpu> unieinfra-dev \
#       bash -lc '/workspace/roy/UnieInfra/benchmark/scripts/ncu_decode_gemv.sh'
#
# Note the `dram__throughput...` metric returns null on this sm_120 part; the
# `gpu__dram_throughput...` spelling is the one that works (E26).
# No `set -u`: sm120_env.sh appends to CPATH/LIBRARY_PATH, which may be unset.
set -eo pipefail

REPO=${REPO:-/workspace/roy/UnieInfra}
NCU=${NCU:-/opt/nvidia/nsight-compute/2025.2.1/ncu}
OUT=${OUT:-$REPO/benchmark/profiles}
SHAPES=${SHAPES:-qkv_proj,o_proj}
BATCHES=${BATCHES:-1,2,4,8}
WITH_VLLM=${WITH_VLLM:---with-vllm}
TAG=${TAG:-e33_decode_gemv}

cd "$REPO"
# shellcheck disable=SC1091
source benchmark/scripts/sm120_env.sh
export TORCHINDUCTOR_COMPILE_THREADS=1

METRICS=(
  gpu__time_duration.sum
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
  dram__bytes.sum
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  lts__t_sector_hit_rate.pct
  sm__warps_active.avg.pct_of_peak_sustained_active
  launch__registers_per_thread
  launch__grid_size
  launch__block_size
  launch__occupancy_limit_registers
  sm__maximum_warps_per_active_cycle_pct
)
IFS=,; METRIC_ARG="${METRICS[*]}"; unset IFS

mkdir -p "$OUT"
"$NCU" \
  --target-processes all \
  --metrics "$METRIC_ARG" \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section LaunchStats \
  --kernel-name-base demangled \
  --cache-control all \
  --clock-control base \
  --export "$OUT/${TAG}" --force-overwrite \
  python3 benchmark/scripts/bench_decode_gemv.py \
      --mode ncu --shapes "$SHAPES" --batches "$BATCHES" $WITH_VLLM

echo "wrote $OUT/${TAG}.ncu-rep"
"$NCU" --import "$OUT/${TAG}.ncu-rep" --page details --csv \
  > "$OUT/${TAG}.csv" 2>/dev/null || true
echo "wrote $OUT/${TAG}.csv"
