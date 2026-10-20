#!/usr/bin/env python3
"""E42: losslessness and schema validity of the draft-masked arm.

Two questions, both of which must come out clean before any throughput number
from E42 means anything:

1. **Is the output still schema-valid?** Both arms constrain the *target*, so
   both should be 100% valid. A drop would mean the draft mask corrupted the
   grammar FSM -- the one real hazard in this change, since it advances the FSM
   and rolls it back.
2. **Is the output IDENTICAL between arms?** Greedy speculative decoding is
   lossless by construction: the drafter only proposes, the target verifies, so
   changing what the drafter proposes must not change a single emitted token.
   This compares the generated text of every request, arm vs arm, at matching
   (k, concurrency, repeat).

Usage: benchmark/e42_validate.py --schema <schema.json> [--prefix e42]

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def load(prefix: str) -> dict:
    """(k, conc, rep, arm) -> (generated_texts, total_output_tokens)."""
    out: dict = {}
    for d in sorted(RESULTS.glob(f"{prefix}_*_k*_r*")):
        parts = d.name.split("_")
        if len(parts) != 4:
            continue
        arm, kstr, rstr = parts[1], parts[2], parts[3]
        for jf in sorted(d.glob("c*.json")):
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            texts = data.get("generated_texts")
            if texts is None:
                continue
            key = (kstr, jf.stem, rstr)
            out.setdefault(key, {})[arm] = (
                texts,
                data.get("total_output_tokens"),
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--prefix", default="e42")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    schema = json.loads(Path(args.schema).read_text())
    cells = load(args.prefix)
    if not cells:
        print("no detailed e42 results found")
        return

    valid_counts: dict = defaultdict(
        lambda: [0, 0, 0]
    )  # arm -> [n, json_ok, schema_ok]
    lines = ["# E42 — validity and losslessness\n"]
    lines.append(
        "## Schema validity (both arms constrain the target, both must be 100%)\n"
    )
    for (_k, _c, _r), arms in cells.items():
        for arm, (texts, _tot) in arms.items():
            stats = valid_counts[arm]
            for t in texts:
                stats[0] += 1
                try:
                    obj = json.loads(t)
                except Exception:
                    continue
                stats[1] += 1
                if check(obj, schema):
                    stats[2] += 1
    lines.append("| arm | responses | valid JSON | schema-conformant |")
    lines.append("|---|---|---|---|")
    for arm in sorted(valid_counts):
        n, j, s = valid_counts[arm]
        lines.append(
            f"| {arm} | {n} | {j} ({100.0 * j / max(n, 1):.1f}%) "
            f"| {s} ({100.0 * s / max(n, 1):.1f}%) |"
        )

    lines.append("\n## Losslessness: identical greedy output, arm vs arm\n")
    lines.append(
        "| k | C | rep | requests | identical texts | total_output_tokens off/on |"
    )
    lines.append("|---|---|---|---|---|---|")
    all_identical = True
    num_pairs = 0
    for kstr, cstr, rstr in sorted(cells):
        arms = cells[(kstr, cstr, rstr)]
        if set(arms) != {"maskoff", "maskon"}:
            continue
        num_pairs += 1
        off, on = arms["maskoff"][0], arms["maskon"][0]
        n = min(len(off), len(on))
        same = sum(1 for i in range(n) if off[i] == on[i])
        if same != n or len(off) != len(on):
            all_identical = False
        lines.append(
            f"| {kstr} | {cstr} | {rstr} | {n} | {same}/{n} "
            f"| {arms['maskoff'][1]} / {arms['maskon'][1]} |"
        )
    if not num_pairs:
        # An empty comparison is not a pass. Say so rather than print "LOSSLESS"
        # off zero rows.
        msg = (
            "NOT TESTED — no (k, C, repeat) cell has BOTH arms, so nothing was "
            "compared. Do not read this as a pass."
        )
    elif all_identical:
        msg = (
            f"LOSSLESS — all {num_pairs} paired cells produced byte-identical "
            "greedy text in both arms."
        )
    else:
        msg = (
            "DIVERGENT — see rows above; investigate before trusting any "
            "throughput number."
        )
    lines.append("\n**Verdict: " + msg + "**\n")
    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(args.out) if args.out else RESULTS.parent / "E42_VALIDITY.md"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
