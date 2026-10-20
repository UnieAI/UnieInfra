#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
#
# ours-vs-vLLM-native A/B for the quantized dispatch paths (E38).
#
# Rotating arms, one process each, N rounds, plus a CONTROL arm that keeps the
# UnieInfra wrapper live but UNREGISTERS the kernel — that arm isolates the cost
# of the dispatch wrapper from the benefit of the kernel, and is the only way to
# read a small delta honestly.
#
#   GPU=2 ./run_quant_ab.sh nvfp4 /workspace/models/nvidia/Qwen3-8B-NVFP4
#   GPU=2 ./run_quant_ab.sh fp8   /workspace/models/Qwen/Qwen3-8B
set -euo pipefail
MODE="${1:?usage: run_quant_ab.sh <fp8|nvfp4> <model>}"
MODEL="${2:?usage: run_quant_ab.sh <fp8|nvfp4> <model>}"
GPU="${GPU:-2}"
ROUNDS="${ROUNDS:-2}"
PY="${PY:-/home/ubuntu/vllm-bench-venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${OUTDIR:-$HERE/../results/vllm-plugin-quant}"
mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
# sm_120 correctness shims, applied identically to BOTH arms.
export VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM's torch.compile/AOT cache is keyed on VllmConfig, not on our env: two
# arms differing only in a UNIEINFRA_* variable would otherwise share a compiled
# graph and report a false 0% (E34).
export VLLM_DISABLE_COMPILE_CACHE=1
export BENCH_MODEL="$MODEL"
export BENCH_CONC="${BENCH_CONC:-1,32,128}"
export BENCH_OLEN="${BENCH_OLEN:-256}"
export BENCH_ILEN="${BENCH_ILEN:-512}"
export BENCH_REPS="${BENCH_REPS:-3}"

if [ "$MODE" = "nvfp4" ]; then
  # vLLM's `auto` NVFP4 backend picks FlashInfer, whose JIT rejects major==12;
  # and the checkpoint's fp8 KV cache makes it pick the FlashInfer attention
  # backend too. Both shims go to both arms.
  export BENCH_LINEAR_BACKEND=cutlass BENCH_ATTN=TRITON_ATTN
  NATIVE_QUANT=modelopt_fp4
  DISABLE_VAR=UNIEINFRA_DECODE_GEMV_NVFP4
else
  NATIVE_QUANT=fp8
  DISABLE_VAR=UNIEINFRA_DECODE_GEMV_FP8
fi

for r in $(seq 1 "$ROUNDS"); do
  BENCH_QUANT=unieinfra UNIEINFRA_VLLM_QUANT="$MODE" \
    BENCH_TAG="${MODE}_ours_r$r" BENCH_OUT="$OUTDIR/${MODE}_ours_r$r.json" \
    UNIEINFRA_KERNEL_SELECTION_DUMP="$OUTDIR/${MODE}_selection_r$r" \
    "$PY" "$HERE/bench_quant_arm.py"
  BENCH_QUANT="$NATIVE_QUANT" \
    BENCH_TAG="${MODE}_vllm_r$r" BENCH_OUT="$OUTDIR/${MODE}_vllm_r$r.json" \
    "$PY" "$HERE/bench_quant_arm.py"
done

# CONTROL: wrapper live, kernel unregistered -> dispatch finds nothing and the
# engine's own scaled GEMM runs. Any delta here is the wrapper, not the kernel.
BENCH_QUANT=unieinfra UNIEINFRA_VLLM_QUANT="$MODE" "$DISABLE_VAR"=0 \
  BENCH_TAG="${MODE}_ctrl" BENCH_OUT="$OUTDIR/${MODE}_ctrl.json" \
  "$PY" "$HERE/bench_quant_arm.py"
