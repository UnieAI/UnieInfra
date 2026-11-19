#!/usr/bin/env python3
"""Matched-interactivity comparison against the vLLM 900 s official reference.

Everything lives in scripts/fair.py so the dashboard, this table and the
experiments log cannot disagree. Flags:
  --md    markdown table (default: plain text)
  --json  machine-readable rows
  --ep    only expert-parallel cells      --tp  only tensor-parallel cells
  --tol F relative interactivity tolerance for a "near" match (default 0.25)
"""
import json
import sys

sys.path.insert(0, "/home/smc/workspace/roy/scripts")
from fair import TOL, fmt, geomean, matched  # noqa: E402


def argval(flag, default):
    if flag in sys.argv:
        return float(sys.argv[sys.argv.index(flag) + 1])
    return default


mode = "ep" if "--ep" in sys.argv else ("tp" if "--tp" in sys.argv else "any")
tol = argval("--tol", TOL)
rows = matched(mode, tol)

if "--json" in sys.argv:
    print(json.dumps({"mode": mode, "tol": tol, "rows": rows, "geomean": geomean(rows)}, default=str))
    sys.exit(0)

g = geomean(rows)
n = sum(1 for r in rows if r["ratio"])
if "--md" in sys.argv:
    print("| 並行度 | UnieInfra RPM | vLLM RPM | 互動性 | 比值 | 狀態 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        ours = f"{r['rpm']:.1f}" if r["rpm"] else "-"
        ratio = f"**{r['ratio']:.2f}×**" if r["ratio"] else "-"
        print(f"| c{r['conc']} | {ours} | {r['ref_rpm']:.1f} | {r['ref_inter']:.0f} | {ratio} | {r['status']} |")
    print()
else:
    for r in rows:
        print(fmt(r))
    print()
print(f"mode={mode} tol={tol:.2f} matched {n}/{len(rows)}" + (f" geomean {g:.3f}x" if g else " (no matched point yet)"))
