#!/usr/bin/env python3
"""E39c: malformed-output rate for the structured-output arms.

`--save-detailed` keeps `generated_texts` in the result JSON. This script parses
every generated string and asks two questions:

  1. Is it JSON at all?              (json.loads succeeds)
  2. Does it satisfy the schema?     (keys/types the schema requires)

Schema checking is deliberately hand-rolled and shallow rather than pulling in
`jsonschema`: the schema under test is small and fixed, and a runtime dependency
added only for a benchmark assertion is not worth it.

Usage: benchmark/e39c_validate.py --schema <schema.json> [--prefix e39c]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "unieinfra"


def check(obj, schema) -> bool:
    """Shallow structural validation: type, required keys, property types."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(obj, dict):
            return False
        for k in schema.get("required", []):
            if k not in obj:
                return False
        for k, sub in schema.get("properties", {}).items():
            if k in obj and not check(obj[k], sub):
                return False
        if schema.get("additionalProperties") is False:
            if set(obj) - set(schema.get("properties", {})):
                return False
        return True
    if t == "string":
        return isinstance(obj, str)
    if t == "integer":
        return isinstance(obj, int) and not isinstance(obj, bool)
    if t == "number":
        return isinstance(obj, (int, float)) and not isinstance(obj, bool)
    return True


def _cost_table(prefix: str) -> list[str]:
    """What the constraint COSTS. Read TPOT, not output tok/s.

    The two arms do not emit the same number of tokens -- the constrained arm
    finishes its object and stops, the free arm burns its whole budget -- so
    output tok/s is not a like-for-like comparison. Per-token cost (TPOT) and
    per-request cost (req/s) are.
    """
    rows: dict = {}
    for d in sorted(RESULTS.glob(f"{prefix}_*_r*")):
        arm = d.name.split("_")[1]
        for jf in d.glob("c*.json"):
            j = json.loads(jf.read_text())
            if j.get("completed"):
                rows.setdefault((j["max_concurrency"], arm), []).append(j)

    def med(v, k):
        return statistics.median([x[k] for x in v])

    out = [
        "",
        "## What the constraint costs",
        "",
        "`output tok/s` is NOT the right column here: the constrained arm emits ~48",
        "tokens per request and the free arm always emits its full 128, so tok/s",
        "compares two different amounts of work. TPOT is the per-token cost of the",
        "grammar mask; req/s is what a caller actually feels.",
        "",
        "| C | arm | req/s | TPOT ms | TTFT ms | out tokens/req |",
        "|---|---|---|---|---|---|",
    ]
    for c in sorted({k[0] for k in rows}):
        for arm in ("free", "json"):
            v = rows.get((c, arm))
            if not v:
                continue
            out.append(
                f"| {c} | {arm} | {med(v, 'request_throughput'):.2f} | "
                f"{med(v, 'mean_tpot_ms'):.2f} | {med(v, 'mean_ttft_ms'):.1f} | "
                f"{med(v, 'total_output_tokens') / med(v, 'completed'):.1f} |"
            )
        f, j2 = rows.get((c, "free")), rows.get((c, "json"))
        if f and j2:
            out.append(
                f"| {c} | **json vs free** | "
                f"**{(med(j2, 'request_throughput') / med(f, 'request_throughput') - 1) * 100:+.1f}%** | "
                f"**{(med(j2, 'mean_tpot_ms') / med(f, 'mean_tpot_ms') - 1) * 100:+.1f}%** | "
                f"**{(med(j2, 'mean_ttft_ms') / med(f, 'mean_ttft_ms') - 1) * 100:+.1f}%** | |"
            )
    out += [
        "",
        "**Caveat, stated because it drives the 0% above:** Qwen3-8B answers in its",
        "default reasoning mode, so the free arm spends the whole 128-token budget",
        "inside `<think>` and the JSON never arrives. The constrained arm cannot emit",
        "`<think>` at all -- the grammar forbids it -- so it answers immediately. This",
        "is therefore a measurement of *schema constraint vs a reasoning model asked",
        "for JSON on a 128-token budget*, a realistic agent setting but NOT a pure",
        "grammar-overhead number. The pure overhead is the TPOT column: +0.0% / +2.7%.",
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--prefix", default="e39c")
    args = ap.parse_args()
    schema = json.loads(Path(args.schema).read_text())

    # (arm, conc) -> list of (parse_ok_rate, schema_ok_rate) per repeat
    agg: dict = {}
    for d in sorted(RESULTS.glob(f"{args.prefix}_*_r*")):
        arm = d.name.split("_")[1]
        for jf in sorted(d.glob("c*.json")):
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            texts = data.get("generated_texts") or []
            if not texts:
                continue
            conc = data.get("max_concurrency")
            n = len(texts)
            parse_ok = schema_ok = 0
            for t in texts:
                s = (t or "").strip()
                # be generous to the unconstrained arm: strip a markdown fence
                if s.startswith("```"):
                    s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                parse_ok += 1
                if check(obj, schema):
                    schema_ok += 1
            agg.setdefault((arm, conc), []).append(
                (100.0 * parse_ok / n, 100.0 * schema_ok / n, n)
            )

    if not agg:
        print("[e39c] no detailed results found yet")
        return

    lines = [
        "# E39c — structured output validity",
        "",
        "Same 256 prompts on both arms; the prompt itself asks for one JSON object,",
        "so the unconstrained arm is being judged on what it was actually told to do.",
        "A leading markdown fence is stripped before parsing, i.e. the free arm is",
        "given the benefit of the doubt.",
        "",
        "| arm | C | valid JSON % | schema-conformant % | n/run |",
        "|---|---|---|---|---|",
    ]
    for (arm, conc), vals in sorted(
        agg.items(), key=lambda kv: (kv[0][1] or 0, kv[0][0])
    ):
        p = statistics.median(v[0] for v in vals)
        s = statistics.median(v[1] for v in vals)
        lines.append(f"| {arm} | {conc} | {p:.1f} | {s:.1f} | {vals[0][2]} |")

    lines += _cost_table(args.prefix)
    out = RESULTS.parent / "E39C_VALIDITY.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
