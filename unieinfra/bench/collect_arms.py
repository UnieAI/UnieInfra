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

"""Read every measured arm off disk and emit one table.

Recovers the cell from the artefact filename -- .arm.sh tags everything
E-<engine>-<model>-<shape>-<conc> -- rather than from a hand-kept list of tags,
which went stale every time a sweep was added.

Parses the .bench text rather than any per-run TSV: an empty metric collapses
under a tab separator and shifts every later column left, which once turned a
goodput of 0.02 into a throughput of 0.02. Label -> value cannot make that
mistake.

Reuse always comes from the engine's own counters, never from what the corpus
claims: a corpus advertising 98% delivered 13.3% once its sessions were
strided.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path("/workspace/roy")
OUT = ROOT / ".results.json"

LABELS = {
    "ok": r"Successful requests",
    "fail": r"Failed requests",
    "dur": r"Benchmark duration \(s\)",
    "rps": r"Request throughput \(req/s\)",
    "gp": r"Request goodput \(req/s\)",
    "tps": r"Output token throughput \(tok/s\)",
    "ttft50": r"Median TTFT \(ms\)",
    "ttft99": r"P99 TTFT \(ms\)",
    "tpot50": r"Median TPOT \(ms\)",
    "tpot99": r"P99 TPOT \(ms\)",
}

TAG = re.compile(r"^E-(?P<engine>ours|vllm)-(?P<model>.+)-(?P<shape>agent-127k|agent-26k|RAG|general)-(?P<conc>\d+)$")


def metrics(path):
    if not path.exists():
        return {}
    text = path.read_bytes().decode("utf-8", "ignore")
    out = {}
    for key, label in LABELS.items():
        m = re.search(label + r":\s*([0-9.]+)", text)
        if m:
            out[key] = float(m.group(1))
    return out


def reuse(log, engine):
    if not log.exists():
        return None
    text = log.read_bytes().decode("utf-8", "ignore")
    if engine == "ours":
        new = sum(int(m.group(1)) for m in re.finditer(r"#new-token: (\d+)", text))
        cached = sum(int(m.group(1)) for m in re.finditer(r"#cached-token: (\d+)", text))
        return 100 * cached / (new + cached) if new + cached else None
    hits = re.findall(r"Prefix cache hit rate: ([0-9.]+)", text)
    return float(hits[-1]) if hits else None


def failure_reason(log):
    """Why a cell produced nothing -- as much a result as a number is."""
    if not log.exists():
        return "no log (never started)"
    text = log.read_bytes().decode("utf-8", "ignore")
    for pattern, reason in (
        (r"exceeds maximum input length|input exceeds", "prompt exceeds the model's clamped max input"),
        (r"peer access is not supported", "non-peer GPU pair"),
        (r"illegal memory access", "illegal memory access in a kernel"),
        (r"CUDA out of memory|torch.OutOfMemoryError", "out of memory"),
        (r"Engine core initialization failed", "engine init failed"),
        (r"No kernel found|NoKernelFoundError", "no kernel for this shape"),
        (r"ModuleNotFoundError|ImportError", "import error"),
        (r"startup failed", "startup failed"),
    ):
        if re.search(pattern, text):
            return reason
    return "ran but wrote no metrics"


def main():
    rows, failures = [], []
    seen = set()
    for bench in sorted(ROOT.glob(".ag-*-E-*.bench")) + sorted(ROOT.glob(".ix*-E-*.bench")):
        # Recover the tag by finding where it starts rather than by slicing a
        # fixed prefix: the vLLM InferenceX harness writes ".ixv-", one
        # character longer than ".ix-", so a fixed slice ate the leading "E"
        # and silently dropped every vLLM general cell from the matrix.
        name = bench.name
        start = name.find("E-")
        if start < 0:
            continue
        m = TAG.match(name[start:].removesuffix(".bench"))
        if not m:
            continue
        key = (m.group("engine"), m.group("model"), m.group("shape"), int(m.group("conc")))
        if key in seen:
            continue
        seen.add(key)
        log = bench.with_suffix(".log")
        met = metrics(bench)
        if not met or not met.get("ok"):
            failures.append((key, failure_reason(log)))
            continue
        engine, model, shape, conc = key
        rows.append({
            "engine": engine, "model": model, "shape": shape, "conc": conc,
            **met,
            "reuse": reuse(log, engine),
            "pass_rate": 100 * met["gp"] / met["rps"] if met.get("rps") else None,
            "per_user": 1000 / met["tpot50"] if met.get("tpot50") else None,
        })

    json.dump(rows, open(OUT, "w"), indent=1)
    hdr = f"{'engine':6} {'model':14} {'shape':11} {'C':>4} {'ok':>4} {'tok/s':>8} {'gp/min':>8} {'pass%':>6} {'TTFTp50':>8} {'TPOTp50':>8} {'reuse%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["model"], r["shape"], r["conc"], r["engine"])):
        print(f"{r['engine']:6} {r['model']:14} {r['shape']:11} {r['conc']:4d} "
              f"{int(r.get('ok', 0)):4d} {r.get('tps', 0):8.1f} {(r.get('gp') or 0) * 60:8.1f} "
              f"{r.get('pass_rate') or 0:6.0f} {r.get('ttft50', 0):8.0f} {r.get('tpot50', 0):8.1f} "
              f"{r.get('reuse') or 0:7.1f}")
    print(f"\n{len(rows)} cells with data -> {OUT}")
    if failures:
        print(f"\n{len(failures)} cells attempted but empty:")
        for (engine, model, shape, conc), why in sorted(failures):
            print(f"  {engine:6} {model:14} {shape:11} c={conc:<4} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
