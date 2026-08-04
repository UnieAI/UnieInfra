#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# End-to-end A/B for measurement-based kernel selection: batch-1 Qwen3-8B decode
# with autotuning ON versus the heuristic, one process per arm.
#
# One process per arm is not a nicety. vLLM captures decode CUDA graphs at engine
# construction, so a flag toggled after startup replays the graph that was
# already captured; the only honest A/B is two engines.
#
# Three arms:
#   heuristic  -- UNIEINFRA_AUTOTUNE unset (today's behaviour)
#   autotune   -- UNIEINFRA_AUTOTUNE=1 with an empty cache (cold: it measures)
#   autotune2  -- UNIEINFRA_AUTOTUNE=1 with the cache the previous arm wrote
#                 (warm: it must reach the same decisions without measuring)
#
# Run INSIDE the `unieinfra-vllm-build` container; the venv only works there.
# PYTHONPATH puts the unieinfra-kernel SOURCE tree ahead of the non-editable
# install in site-packages, so live edits take effect without mutating anyone
# else's environment.
set -uo pipefail

GPU="${GPU:-3}"
MODEL="${MODEL:-/workspace/models/Qwen/Qwen3-8B}"
REPS="${REPS:-3}"
REPO=/workspace/roy/UnieInfra
OUT="${REPO}/benchmark/results/autotune/e2e"
CACHE="${REPO}/benchmark/results/autotune/e2e_cache"
mkdir -p "${OUT}"
rm -rf "${CACHE}"

arm() {
  local label="$1"; shift
  echo "=== ${label} ==="
  local t0 t1
  t0=$(date +%s.%N)
  env "$@" \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    PYTHONPATH="${REPO}/unieinfra-kernel" \
    UNIEINFRA_KERNEL_AUTOTUNE_CACHE="${CACHE}" \
    python "${REPO}/benchmark/scripts/bench_vllm_batch1_decode.py" \
      --model "${MODEL}" --reps "${REPS}" --label "${label}" \
      --gpu-mem-util "${GPU_MEM_UTIL:-0.55}" \
      --json "${OUT}/${label}.json" > "${OUT}/${label}.log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)
  echo "${label} rc=${rc} wall=$(python -c "print(f'{${t1} - ${t0}:.1f}')")s"
  grep -E "UnieInfra autotune|autotune:" "${OUT}/${label}.log" | head -20
  grep -o 'UNIEINFRA_RESULT .*' "${OUT}/${label}.log" \
    | python -c "import sys,json;d=json.loads(sys.stdin.read().split(' ',1)[1]);print('  decode tok/s:',d['decode_tok_s_samples'],'median',d['decode_tok_s_median'])" \
    2>/dev/null || echo "  (no result line)"
  echo
}

arm heuristic UNIEINFRA_AUTOTUNE=0
arm autotune   UNIEINFRA_AUTOTUNE=1
arm autotune2  UNIEINFRA_AUTOTUNE=1

echo "cache file(s):"
ls -l "${CACHE}" 2>/dev/null
