#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
#
# E44 step 2: hardware-counter audit of vLLM's steady-state decode kernels at
# high concurrency, using the metric set E26/E33 used so the numbers are
# directly comparable across engines.
#
# Counters are locked by the driver's RmProfilingAdminOnly plus a missing
# CAP_SYS_ADMIN, so this must run inside a *privileged* exec session. That is
# non-destructive: it does not restart the container and does not touch other
# GPUs or other users' jobs (E26).
#
#   docker exec --privileged -e CUDA_VISIBLE_DEVICES=<free gpu> \
#       unieinfra-vllm-build bash -lc \
#       '/workspace/roy/UnieInfra/benchmark/scripts/e44_ncu_decode.sh'
#
# `dram__throughput...` returns null on this sm_120 part; the
# `gpu__dram_throughput...` spelling is the one that works (E26).
set -eo pipefail

REPO=${REPO:-/workspace/roy/UnieInfra}
NCU=${NCU:-/opt/nvidia/nsight-compute/2025.2.1/ncu}
PY=${PY:-/workspace/roy/venvs/unieinfra-vllm-venv/bin/python}
OUT=${OUT:-$REPO/benchmark/profiles/e44}
TAG=${TAG:-e44_c128}
C=${C:-128}
ILEN=${ILEN:-1024}
COUNT=${COUNT:-30}
KERNELS=${KERNELS:-regex:(flash_fwd_splitkv_kernel|s16816gemm|s161616gemm|splitKreduce)}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}
MAXLEN=${MAXLEN:-1152}

cd "$REPO"
export TORCHINDUCTOR_COMPILE_THREADS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export FLASHINFER_DISABLE_VERSION_CHECK=1

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
echo "[e41] ncu -> $OUT/${TAG}.ncu-rep  (C=$C ilen=$ILEN count=$COUNT)"
"$NCU" \
  --target-processes all \
  --profile-from-start off \
  --kernel-name "$KERNELS" \
  --launch-count "$COUNT" \
  --metrics "$METRIC_ARG" \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section LaunchStats \
  --kernel-name-base demangled \
  --cache-control all \
  --clock-control base \
  --export "$OUT/${TAG}" --force-overwrite \
  "$PY" benchmark/scripts/e44_ncu_harness.py \
      --concurrency "$C" --ilen "$ILEN" --olen 3 \
      --gpu-mem-util "$GPU_MEM_UTIL" --max-model-len "$MAXLEN" \
      --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}" \
      ${ATTENTION_BACKEND:+--attention-backend "$ATTENTION_BACKEND"}

echo "[e41] wrote $OUT/${TAG}.ncu-rep"
