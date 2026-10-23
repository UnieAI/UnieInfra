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

"""Tick off experiment.md from the measured rows, and keep the numbers.

A checklist that only records "done" loses the reason the cell was worth
running. Each ticked line therefore carries the goodput, throughput and TTFT
it produced, and every UnieInfer line carries the verdict against vLLM in the
same cell -- those losses are the optimisation queue, so they have to be
visible in the checklist rather than in a separate report that drifts.

Reads /workspace/roy/.results.json (written by the collector) and rewrites
experiment.md in place. Re-runnable: it rebuilds every line from the data, so
a re-measured arm overwrites its old verdict instead of appending a second one.
"""

import json
import pathlib
import re
import sys

RESULTS = pathlib.Path("/workspace/roy/.results.json")
TABLE = pathlib.Path(__file__).resolve().parents[1] / "experiment.md"

# experiment.md's vocabulary -> the collector's
SHAPES = {
    "agent 127k": "agent-127k",
    "agent short": "agent-26k",
    "RAG": "RAG",
    "General": "general",
}
MODELS = {
    "qwen3-27b": "27B",
    "qwen3.6-35b-a3b": "35B-A3B",
    "qwen3.5-122b-a10b-fp8": "122B-A10B-FP8",
}
ENGINES = {"vllm": "vllm", "UnieInfer": "ours"}

# Anything after the model is a summary this script appended on an earlier
# run; match the stem and ignore the tail, or the second run sees no cells.
LINE = re.compile(
    r"^\[.?\]\s*(?P<engine>vllm|UnieInfer)\s+c=(?P<conc>\d+),\s*"
    r"(?P<shape>[^,]+?)\s*,\s*(?P<model>[^\s]+)"
)


def index(rows):
    return {(r["engine"], r["model"], r["shape"], r["conc"]): r for r in rows}


def summarise(r):
    """The three numbers that decide whether a cell is good, in one line."""
    gp = (r.get("gp") or 0) * 60
    tps = r.get("tps") or 0
    ttft = r.get("ttft50") or 0
    pas = r.get("pass_rate") or 0
    return f"{gp:.1f}/min, {tps:.0f} tok/s, TTFT p50 {ttft:.0f} ms, 合格 {pas:.0f}%"


def verdict(ours, theirs):
    """Ours against vLLM in the same cell, on the goodput that is the metric.

    The bar is the repository's own: +10% against the other frameworks, parity
    is not a result (AGENTS.local.md, step 3). So a cell only passes at 1.10x;
    anything at or below parity is a loss, and the band between them is a cell
    that runs but has not earned anything yet. Both of the latter are work.
    """
    if ours is None or theirs is None:
        return None
    a = (ours.get("gp") or 0) * 60
    b = (theirs.get("gp") or 0) * 60
    if b <= 0 and a <= 0:
        return "LOSS 兩邊都是 0 <- 待優化"
    if b <= 0:
        return f"WIN {a:.1f} vs 0"
    ratio = a / b
    if ratio >= 1.10:
        return f"WIN {ratio:.2f}x ({a:.1f} vs {b:.1f})"
    if ratio < 1.0:
        return f"LOSS {ratio:.2f}x ({a:.1f} vs {b:.1f}) <- 待優化"
    return f"PARITY {ratio:.2f}x ({a:.1f} vs {b:.1f}) <- 未達 +10%，仍算待優化"


def main():
    if not RESULTS.exists():
        print(f"no results at {RESULTS}", file=sys.stderr)
        return 1
    by_key = index(json.load(open(RESULTS)))

    out, done, lost, todo = [], 0, [], 0
    for raw in TABLE.read_text().splitlines():
        m = LINE.match(raw.strip())
        if not m:
            out.append(raw)
            continue
        engine = ENGINES[m.group("engine")]
        shape = SHAPES.get(m.group("shape").strip())
        model = MODELS.get(m.group("model").strip())
        conc = int(m.group("conc"))
        stem = f"{m.group('engine')} c={conc}, {m.group('shape').strip()} , {m.group('model').strip()}"

        row = by_key.get((engine, model, shape, conc)) if shape and model else None
        if row is None:
            out.append(f"[] {stem}")
            todo += 1
            continue

        done += 1
        line = f"[x] {stem}  -- {summarise(row)}"
        if engine == "ours":
            v = verdict(row, by_key.get(("vllm", model, shape, conc)))
            if v:
                line += f"  [{v}]"
                if v.startswith(("LOSS", "PARITY")):
                    lost.append(f"{model} / {shape} / c={conc}: {v}")
        out.append(line)

    TABLE.write_text("\n".join(out) + "\n")
    print(f"{done} cells measured, {todo} outstanding")
    if lost:
        print(
            f"\n{len(lost)} cells not yet passing the +10% bar -- the optimisation queue:"
        )
        for item in lost:
            print(f"  {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
