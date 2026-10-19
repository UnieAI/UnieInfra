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

"""Median AgentX duration per engine and cell over repeated arms.

    python summarize.py RESULTS_DIR ENGINE=TAG[,TAG...] [ENGINE=TAG,...]

The first engine is the baseline the speedups are computed against. A cell
with failed requests is reported, never silently averaged.
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


if __name__ == "__main__":
    main()
