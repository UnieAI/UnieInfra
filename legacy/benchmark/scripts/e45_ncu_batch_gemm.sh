#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
#
# E45: hardware-counter comparison of the incumbent cuBLASLt/CUTLASS decode
# projection kernels against the first-party decode-batch GEMM, using the same
# metric set E26/E33/E44 used so the numbers are directly comparable.
#
# The metrics that carry the argument are the *launch statistics* --
# `launch__grid_size`, `launch__waves_per_multiprocessor`,
# `sm__maximum_warps_per_active_cycle_pct` -- because those are properties of
# the launch configuration and are insensitive to whether another agent's engine
# is sharing the card. The throughput percentages are not, so read them with
# that caveat and prefer `--mode compare` in the Python harness for timing.
#
# Counters are locked by the driver's RmProfilingAdminOnly plus a missing
# CAP_SYS_ADMIN, so this must run inside a *privileged* exec session. That is
# non-destructive: it does not restart the container and does not touch other
# GPUs or other users' jobs (E26).
#
#   docker exec --privileged -e CUDA_VISIBLE_DEVICES=<free gpu> \
#       unieinfra-vllm-build bash -lc \
#       '/workspace/roy/UnieInfra/benchmark/scripts/e45_ncu_batch_gemm.sh'
#
# `dram__throughput...` returns null on this sm_120 part; the
# `gpu__dram_throughput...` spelling is the one that works (E26).
# No `set -u`: the venv activate script touches unset variables.
set -eo pipefail

REPO=${REPO:-/workspace/roy/UnieInfra}
NCU=${NCU:-/opt/nvidia/nsight-compute/2025.2.1/ncu}
PY=${PY:-/workspace/roy/venvs/unieinfra-vllm-venv/bin/python}
OUT=${OUT:-$REPO/benchmark/profiles/e45}
TAG=${TAG:-e45_batch_gemm}
SHAPES=${SHAPES:-gate_up_proj,down_proj,qkv_proj,o_proj}
BATCHES=${BATCHES:-128}

cd "$REPO"
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/e45_triton}

METRICS=(
  gpu__time_duration.sum
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
  dram__bytes.sum
  dram__bytes_read.sum
  dram__bytes_write.sum
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  lts__t_sector_hit_rate.pct
  sm__warps_active.avg.pct_of_peak_sustained_active
  launch__registers_per_thread
  launch__grid_size
  launch__block_size
  launch__waves_per_multiprocessor
  launch__occupancy_limit_registers
  launch__occupancy_limit_shared_mem
  sm__maximum_warps_per_active_cycle_pct
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
)
IFS=,; METRIC_ARG="${METRICS[*]}"; unset IFS

mkdir -p "$OUT"
echo "[e45] ncu -> $OUT/${TAG}.ncu-rep  (shapes=$SHAPES batches=$BATCHES)"
"$NCU" \
  --target-processes all \
  --profile-from-start off \
  --metrics "$METRIC_ARG" \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section LaunchStats \
  --kernel-name-base demangled \
  --cache-control all \
  --clock-control base \
  --export "$OUT/${TAG}" --force-overwrite \
  "$PY" benchmark/scripts/bench_decode_batch_gemm.py \
      --mode ncu --shapes "$SHAPES" --batches "$BATCHES" --reserve-gb 0

"$NCU" --import "$OUT/${TAG}.ncu-rep" --csv --page details > "$OUT/${TAG}.csv"
echo "[e45] wrote $OUT/${TAG}.ncu-rep and .csv"
