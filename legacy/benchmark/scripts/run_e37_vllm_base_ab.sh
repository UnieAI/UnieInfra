#!/usr/bin/env bash
# E37: does E34's kernel win reproduce on the NEW BASE, through the DIRECT seam?
#
# E34 ran UnieInfra's kernel selection layer inside stock vLLM 0.25.1 via the
# plugin API (`vllm.general_plugins` -> `register_quantization_config` ->
# `--quantization unieinfra`) and measured batch-1 Qwen3-8B decode at
# 84.84 -> 88.52 tok/s (+4.34%) and 85.19 -> 88.68 (+4.10%).
#
# The base decision has since changed: UnieInfra owns the engine source
# (`unieinfra-vllm/`, a hard fork of vLLM @ d02df748), so the kernel layer is
# wired DIRECTLY into `dispatch_unquantized_gemm()` and there is no
# `--quantization` flag. This script re-runs the same acceptance test through
# that direct seam.
#
# Protocol, copied from E34 because the traps are the same:
#   * ONE PROCESS PER ARM. vLLM captures decode CUDA graphs at engine
#     construction, so flipping an env var after startup replays the graph that
#     was already captured and measures nothing.
#   * VLLM_DISABLE_COMPILE_CACHE=1 on every arm. vLLM's compile/AOT cache is
#     keyed on `VllmConfig`, NOT on env vars, so two runs differing only in
#     UNIEINFRA_GEMM would otherwise share ~/.cache/vllm/torch_compile_cache/
#     and the second would silently replay the first's graph. This bit E34.
#   * ROTATED A,B,A,B rather than A,A,B,B, so slow drift in box conditions is
#     spread across arms instead of being confounded with one.
#
# Arms:
#   stock  UNIEINFRA_GEMM=0 -> upstream's `default_unquantized_gemm` (F.linear)
#   unie   UNIEINFRA_GEMM=1 -> `unieinfra_unquantized_gemm` -> the registry
#
# Usage: run_e37_vllm_base_ab.sh [PAIRS]     (default 2)
set -euo pipefail

REPO=/workspace/roy/UnieInfra
CONTAINER="${CONTAINER:-unieinfra-vllm-build}"
GPU="${GPU:-1}"
PAIRS="${1:-2}"
OUT="$REPO/benchmark/results/vllm-base"
mkdir -p "$OUT"

run() {  # run <arm> <rep>
  local arm="$1" rep="$2" gemm label
  [ "$arm" = "unie" ] && gemm=1 || gemm=0
  label="${arm}-r${rep}"
  echo "=== $label (UNIEINFRA_GEMM=$gemm) $(date -u +%FT%TZ) ==="
  docker exec "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU
    export UNIEINFRA_GEMM=$gemm
    export VLLM_DISABLE_COMPILE_CACHE=1
    export UNIEINFRA_KERNEL_SELECTION_DUMP=$OUT/$label.sel
    export VLLM_LOGGING_LEVEL=INFO
    cd $REPO
    python3 benchmark/scripts/bench_vllm_batch1_decode.py \
      --label $label --json $OUT/$label.json
  " > "$OUT/$label.log" 2>&1 || { echo "ARM FAILED: $label"; tail -30 "$OUT/$label.log"; return 1; }
  grep -o 'UNIEINFRA_RESULT .*' "$OUT/$label.log" | head -1
}

for r in $(seq 1 "$PAIRS"); do
  run stock "$r"
  run unie  "$r"
done

echo
echo "=== summary ==="
python3 - "$OUT" <<'PY'
import glob, json, os, sys, statistics
out = sys.argv[1]
by = {}
for path in sorted(glob.glob(os.path.join(out, "*-r*.json"))):
    if ".sel." in path:      # per-process selection dumps, not arm results
        continue
    d = json.load(open(path))
    by.setdefault(d["label"].split("-r")[0], []).append(d["decode_tok_s_median"])
for arm, vals in by.items():
    print(f"{arm:6s} n={len(vals)} median={statistics.median(vals):.2f} vals={[round(v,2) for v in vals]}")
if {"stock", "unie"} <= by.keys():
    a, b = statistics.median(by["stock"]), statistics.median(by["unie"])
    print(f"delta: {a:.2f} -> {b:.2f} tok/s  ({100*(b-a)/a:+.2f}%)")
PY
