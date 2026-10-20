#!/usr/bin/env python3
"""Compare two AgentX sweep outputs as arms of an A/B.

Two things decide whether an arm is a faster engine or a broken one:

  * throughput and latency, computed from the per-request rows, so the number
    is the same arithmetic the reference tables use;
  * whether the same greedy request produced the same tokens. A batch that
    mixes prefill with decode, or a leaf that slices a packed batch wrongly, can
    drop work and still look faster -- the only cheap evidence that it did not
    is temperature-0 output agreement against an arm whose correctness is
    already trusted.

Rows are matched by request index ``i``, which is the prompt's position in the
corpus, so two cells driven by the same corpus compare request by request.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys


def load(path):
    doc = json.load(open(path))
    rows = {r["i"]: r for r in doc["rows"]}
    return doc, rows


def pct(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summary(doc):
    rows = doc["rows"]
    dur = doc["duration_s"]
    toks = sum(r["ntok"] for r in rows)
    ttfts = [r["ttft"] for r in rows if r["ttft"]]
    tpots = [r["tpot"] for r in rows if r["tpot"]]
    return {
        "n": len(rows),
        "err": sum(1 for r in rows if r["err"]),
        "tok_s": toks / dur if dur else float("nan"),
        "rpm": 60.0 * len(rows) / dur if dur else float("nan"),
        "tok_per_req": toks / len(rows) if rows else float("nan"),
        "ttft50": pct(ttfts, 0.50),
        "ttft95": pct(ttfts, 0.95),
        "tpot50": pct(tpots, 0.50),
        "tpot90": pct(tpots, 0.90),
        "dur": dur,
    }


def first_divergence(a: str, b: str) -> int:
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("test")
    ap.add_argument(
        "--prefix-chars",
        type=int,
        default=24,
        help="characters of a diverging completion to print",
    )
    ap.add_argument(
        "--min-agreement",
        type=float,
        default=0.95,
        help="exit non-zero below this exact-match rate",
    )
    args = ap.parse_args()

    db, rb = load(args.base)
    dt, rt = load(args.test)
    sb, st = summary(db), summary(dt)

    print(
        f"{'arm':<10}{'tok/s':>9}{'rpm':>8}{'tok/req':>9}{'ttft50':>9}"
        f"{'ttft95':>9}{'tpot50':>8}{'tpot90':>8}{'err':>5}"
    )
    for name, s in (("base", sb), ("test", st)):
        print(
            f"{name:<10}{s['tok_s']:>9.1f}{s['rpm']:>8.1f}"
            f"{s['tok_per_req']:>9.1f}{s['ttft50']:>9.0f}{s['ttft95']:>9.0f}"
            f"{s['tpot50']:>8.0f}{s['tpot90']:>8.0f}{s['err']:>5}"
        )
    ratio = st["tok_s"] / sb["tok_s"] if sb["tok_s"] else float("nan")
    print(f"\ntest/base throughput: {ratio:.3f}x")

    common = sorted(set(rb) & set(rt))
    same = 0
    diffs = []
    for i in common:
        a, b = rb[i]["text"], rt[i]["text"]
        if a == b:
            same += 1
        else:
            diffs.append((i, a, b))
    rate = same / len(common) if common else float("nan")
    print(
        f"greedy agreement: {same}/{len(common)} = {rate:.4f}"
        f" (rows only in base: {len(set(rb) - set(rt))},"
        f" only in test: {len(set(rt) - set(rb))})"
    )

    for i, a, b in diffs[:5]:
        k = first_divergence(a, b)
        print(
            f"\n  request {i} diverges at char {k}"
            f" (base {len(a)} chars, test {len(b)} chars)"
        )
        print(f"    base: ...{a[max(0, k - 20):k + args.prefix_chars]!r}")
        print(f"    test: ...{b[max(0, k - 20):k + args.prefix_chars]!r}")

    lengths = [(rb[i]["ntok"], rt[i]["ntok"]) for i in common]
    short = [(i, x, y) for i, (x, y) in zip(common, lengths) if x != y]
    if short:
        print(f"\ntoken counts differ on {len(short)} requests" f" (first: {short[0]})")

    if rate < args.min_agreement:
        print(f"\nFAIL: agreement {rate:.4f} < {args.min_agreement}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
