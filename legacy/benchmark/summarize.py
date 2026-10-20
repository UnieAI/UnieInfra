#!/usr/bin/env python3
"""Summarize `vllm bench serve` raw result JSONs into one comparable CSV.

Policy: save WHATEVER the benchmark produced. The CSV keeps *every* scalar field
present across the runs (union of keys), one row per concurrency, so nothing is
dropped. Nested/array fields (e.g. per-request latency lists) stay in the raw
JSONs untouched — this script only reads them.

Usage: summarize.py <raw_dir> <out_csv> <engine_name>
"""

import csv
import glob
import json
import os
import sys

raw_dir, out_csv, engine = sys.argv[1], sys.argv[2], sys.argv[3]

# A few keys we like to see first / use for sorting + the compact table.
PREFERRED = [
    "engine",
    "max_concurrency",
    "num_prompts",
    "completed",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "mean_e2el_ms",
    "p99_e2el_ms",
]

rows = []
all_keys = set()
for path in sorted(glob.glob(os.path.join(raw_dir, "c*.json"))):
    if not path.endswith(".json"):
        continue
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"[summarize] skip {path}: {e}", file=sys.stderr)
        continue
    row = {"engine": engine}
    for k, v in d.items():
        # keep every scalar the benchmark emitted; skip lists/dicts (raw has them)
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[k] = round(v, 4) if isinstance(v, float) else v
    rows.append(row)
    all_keys.update(row.keys())

# column order: preferred keys first (those that exist), then everything else sorted
ordered = [k for k in PREFERRED if k in all_keys]
ordered += sorted(k for k in all_keys if k not in ordered)

rows.sort(key=lambda r: (r.get("max_concurrency") or 0))

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

# derive per-user decode rate (tok/s/user = 1000/TPOT) — the "does per-user speed drop?" metric
for r in rows:
    tp = r.get("mean_tpot_ms")
    r["tok_s_per_user"] = (
        round(1000.0 / tp, 1) if isinstance(tp, (int, float)) and tp else ""
    )

# compact human table for the headline metrics
show = [
    c
    for c in [
        "max_concurrency",
        "request_throughput",
        "output_throughput",
        "tok_s_per_user",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
    ]
    if c in all_keys or c == "tok_s_per_user"
]
print(
    f"engine={engine}  ({len(rows)} concurrency points, {len(ordered)} metrics/row saved)"
)
print("  ".join(f"{c:>16}" for c in show))
for r in rows:
    print("  ".join(f"{str(r.get(c, '')):>16}" for c in show))
