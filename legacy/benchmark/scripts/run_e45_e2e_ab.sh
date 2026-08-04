#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
#
# E45: does the decode-batch GEMM's micro-benchmark win survive end to end?
#
# Protocol, inherited from E34/E37 because the traps are the same:
#   * ONE PROCESS PER ARM. vLLM captures decode CUDA graphs at engine
#     construction, so flipping an env var after startup replays the graph that
#     was already captured and measures nothing.
#   * VLLM_DISABLE_COMPILE_CACHE=1 on every arm. vLLM's compile/AOT cache is
#     keyed on `VllmConfig`, NOT on env vars, so two runs differing only in
#     UNIEINFRA_DECODE_BATCH_GEMM would otherwise share the compile cache and
#     the second would silently replay the first's graph. This bit E34.
#   * ROTATED A,B,A,B rather than A,A,B,B, so slow drift in box conditions is
#     spread across arms instead of being confounded with one.
#
# Both arms run UNIEINFRA_GEMM=1, i.e. through the identical dispatch seam; the
# `off` arm's kernel declines at run time and delegates to F.linear. The arms
# therefore differ in the kernel body and nothing else.
#
# Usage: run_e45_e2e_ab.sh [PAIRS]   (default 2)
set -euo pipefail

REPO=/workspace/roy/UnieInfra
CONTAINER="${CONTAINER:-unieinfra-vllm-build}"
GPU="${GPU:?set GPU=<index of a free GPU>; never GPU 0}"
PAIRS="${1:-2}"
C="${C:-128}"
ILEN="${ILEN:-1024}"
OLEN="${OLEN:-128}"
OUT="$REPO/benchmark/results/e45"
mkdir -p "$OUT"

run() {  # run <arm> <rep>
  local arm="$1" rep="$2" flag label
  [ "$arm" = "on" ] && flag=1 || flag=0
  label="${arm}-r${rep}"
  echo "=== $label (UNIEINFRA_DECODE_BATCH_GEMM=$flag) $(date -u +%FT%TZ) ==="
  docker exec "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU
    # The venv has a NON-editable copy of unieinfra_kernel installed, so without
    # this the engine imports site-packages and never sees the kernel under
    # test -- the first run of this A/B measured two identical arms and only the
    # [select] probe caught it. Prepending the source tree is scoped to this
    # process, unlike reinstalling into a venv other agents are also using.
    export PYTHONPATH=$REPO/unieinfra-kernel:\${PYTHONPATH:-}
    export UNIEINFRA_GEMM=1
    export UNIEINFRA_DECODE_BATCH_GEMM=$flag
    export VLLM_DISABLE_COMPILE_CACHE=1
    export TORCHINDUCTOR_COMPILE_THREADS=1
    export TRITON_CACHE_DIR=/tmp/e45_triton
    export VLLM_LOGGING_LEVEL=ERROR
    source /workspace/roy/venvs/unieinfra-vllm-venv/bin/activate
    cd $REPO
    python benchmark/scripts/e45_e2e_batch_gemm_ab.py \
      --concurrency $C --ilen $ILEN --olen $OLEN \
      --label $label --json $OUT/e2e.jsonl
  " > "$OUT/$label.log" 2>&1 || { echo "ARM FAILED: $label"; tail -40 "$OUT/$label.log"; return 1; }
  grep -o '\[e45-e2e\] RESULT .*' "$OUT/$label.log" | head -1
  grep -o '\[select\] .*' "$OUT/$label.log" | head -4
}

for r in $(seq 1 "$PAIRS"); do
  run off "$r"
  run on  "$r"
done

echo
echo "=== summary ($OUT/e2e.jsonl) ==="
python3 - "$OUT/e2e.jsonl" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["arm"], []).append(r["median_tok_s"])
tot = {}
for r in rows:
    tot.setdefault(r["arm"], []).append(r.get("median_total_tok_s", float("nan")))
for arm in ("off", "on"):
    if arm in by:
        print(f"{arm:4s} n={len(by[arm])} decode={statistics.median(by[arm]):8.2f} "
              f"tok/s  total={statistics.median(tot[arm]):8.2f} tok/s  "
              f"all={[round(x,1) for x in by[arm]]}")
if "off" in by and "on" in by:
    a, b = statistics.median(by["off"]), statistics.median(by["on"])
    ta, tb = statistics.median(tot["off"]), statistics.median(tot["on"])
    print(f"delta: decode {100 * (b - a) / a:+.2f}%   total {100 * (tb - ta) / ta:+.2f}%")
PY
