# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Median AgentX metrics per engine and cell over repeated arms.

    python summarize.py RESULTS_DIR ENGINE=TAG[,TAG...] [ENGINE=TAG,...]

The first engine is the baseline the speedups are computed against. A cell
with failed requests is reported, never silently averaged.

Every request generates exactly OUTPUT_TOKENS tokens (run_cells.sh sets
min_tokens = max_tokens = 128), so output throughput and TPOT come from that
count, not from the driver's ``ntok`` (SSE chunks: speculative decoding
streams several tokens per chunk). TTFT is client-side (queueing included).
Each metric is computed per arm, then the median over the arms is reported.
"""

import json
import statistics
import sys
from pathlib import Path


def durations(results: Path, tags: list[str], conc: int) -> list[float]:
    values = []
    for tag in tags:
        path = results / f"{tag}_c{conc}.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        failed = sum(1 for row in run["rows"] if row.get("err"))
        if failed:
            raise SystemExit(f"{path}: {failed} failed requests")
        values.append(run["duration_s"])
    return values


OUTPUT_TOKENS = 128


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def cell_metrics(path: Path) -> dict[str, float]:
    run = json.loads(path.read_text())
    rows = run["rows"]
    duration = run["duration_s"]
    ttft = [row["ttft"] for row in rows]
    e2e = [row["ms"] for row in rows]
    tpot = [(row["ms"] - row["ttft"]) / (OUTPUT_TOKENS - 1) for row in rows]
    return {
        "duration (s)": duration,
        "RPM": len(rows) / duration * 60,
        "output TPS": len(rows) * OUTPUT_TOKENS / duration,
        "TTFT mean (ms)": statistics.mean(ttft),
        "TTFT p50 (ms)": percentile(ttft, 0.5),
        "TTFT p90 (ms)": percentile(ttft, 0.9),
        "TPOT mean (ms)": statistics.mean(tpot),
        "E2E p50 (ms)": percentile(e2e, 0.5),
        "E2E p90 (ms)": percentile(e2e, 0.9),
    }


def metrics_table(results: Path, engines: list[list[str]], concs) -> None:
    for conc in concs:
        per_engine = {}
        for name, tags in engines:
            runs = []
            for tag in tags.split(","):
                path = results / f"{tag}_c{conc}.json"
                if path.exists():
                    durations(results, [tag], conc)  # refuses failed cells
                    runs.append(cell_metrics(path))
            per_engine[name] = {
                key: statistics.median(run[key] for run in runs) for key in runs[0]
            }
        names = [name for name, _ in engines]
        print(f"\n**c{conc}** (median over arms)\n")
        print("| metric | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for key in per_engine[names[0]]:
            cells = [f"{per_engine[name][key]:.1f}" for name in names]
            print(f"| {key} | " + " | ".join(cells) + " |")


def main() -> None:
    results = Path(sys.argv[1])
    engines = [arg.split("=", 1) for arg in sys.argv[2:]]
    concs = (8, 16, 32)
    medians = {}
    print("| engine | runs | " + " | ".join(f"c{c} (s)" for c in concs) + " |")
    print("|---|---|" + "---|" * len(concs))
    for name, tags in engines:
        cells = []
        for conc in concs:
            values = durations(results, tags.split(","), conc)
            medians[name, conc] = statistics.median(values) if values else None
            spread = (
                f" [{min(values):.1f}-{max(values):.1f}]" if len(values) > 1 else ""
            )
            cells.append(f"{medians[name, conc]:.1f}{spread}" if values else "-")
        print(f"| {name} | {len(tags.split(','))} | " + " | ".join(cells) + " |")
    base = engines[0][0]
    for name, _ in engines[1:]:
        ratios = [
            f"c{c} {medians[base, c] / medians[name, c]:.2f}x"
            for c in concs
            if medians[base, c] and medians[name, c]
        ]
        print(f"\n{name} vs {base}: " + ", ".join(ratios))
    metrics_table(results, engines, concs)


if __name__ == "__main__":
    main()
