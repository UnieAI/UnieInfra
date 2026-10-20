# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Goodput as a function of the deadline, for every measured cell.

An SLO is not one number. Reporting a single threshold hides whether an
advantage is structural or lives in a narrow window: on 27B random traffic the
two engines are 7.5x apart at a 2-second deadline and exactly equal at 5
seconds, because their TTFT distributions are about a second apart and
otherwise identical. Sweeping the deadline shows which it is.

Concurrency is the second axis and it moves the curve, so each cell gets its
own line rather than one line per scenario. Output is keyed
model / shape / engine / concurrency for the page to render as small multiples.

Reads the per-request latencies vllm bench serve writes with --save-detailed.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path("/workspace/roy")
OUT = ROOT / ".slo_curves.json"

# Deadline grid, log-spaced from a quarter second to a minute and a half.
GRID = [
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    4000,
    5000,
    7500,
    10000,
    15000,
    20000,
    30000,
    45000,
    60000,
    90000,
    150000,
]

# The TPOT bound stays fixed per shape while the TTFT deadline sweeps; agent
# prompts are two orders of magnitude longer, so they carry a looser budget.
TPOT_BOUND = {"agent-127k": 200, "agent-26k": 200, "RAG": 100, "general": 100}
REPORTED = {"agent-127k": 10000, "agent-26k": 10000, "RAG": 5000, "general": 5000}

TAG = re.compile(
    r"E-(?P<engine>ours|vllm)-(?P<model>.+?)-(?P<shape>agent-127k|agent-26k|RAG|general)-(?P<conc>\d+)"
)


def per_request(path):
    """(ttft_ms, mean_itl_ms) per request, tolerant of second-vs-ms units."""
    d = json.load(open(path))
    out = []
    for ttft, itl in zip(d.get("ttfts", []), d.get("itls", [])):
        if not itl:
            continue
        t = ttft * 1000 if ttft < 100 else ttft
        m = sum(itl) / len(itl)
        out.append((t, m * 1000 if m < 1 else m))
    return out, d.get("duration") or 0, d.get("completed") or 0


def collect():
    files = list(ROOT.glob(".detail/*E-*.json"))
    files += list(ROOT.glob(".ix-E-*.result/*.json"))
    files += list(ROOT.glob(".ixv-E-*.result/*.json"))

    curves = {}
    for path in sorted(files):
        m = TAG.search(str(path))
        if not m:
            continue
        engine, model, shape, conc = (
            m.group("engine"),
            m.group("model"),
            m.group("shape"),
            int(m.group("conc")),
        )
        try:
            reqs, dur, n = per_request(path)
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"skip {path.name}: {exc}", file=sys.stderr)
            continue
        if not reqs or dur <= 0:
            continue
        bound = TPOT_BOUND[shape]
        pts = [
            [t, round(60 * sum(1 for a, b in reqs if a <= t and b <= bound) / dur, 2)]
            for t in GRID
        ]
        curves.setdefault(model, {}).setdefault(shape, {}).setdefault(engine, {})[
            conc
        ] = {
            "pts": pts,
            "n": n,
            "dur": round(dur, 1),
            "ttft_p50": round(sorted(a for a, _ in reqs)[len(reqs) // 2]),
            "tpot_p50": round(sorted(b for _, b in reqs)[len(reqs) // 2], 1),
        }
    return curves


def main():
    curves = collect()
    payload = {
        "tpot_bound": TPOT_BOUND,
        "reported_deadline": REPORTED,
        "grid": GRID,
        "curves": curves,
    }
    json.dump(payload, open(OUT, "w"), ensure_ascii=False)

    total = 0
    for model, shapes in sorted(curves.items()):
        for shape, engines in sorted(shapes.items()):
            got = {e: sorted(c) for e, c in sorted(engines.items())}
            total += sum(len(v) for v in got.values())
            print(
                f"{model:14} {shape:11} "
                + "  ".join(f"{e}={v}" for e, v in got.items())
            )
    print(f"\n{total} curves -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
