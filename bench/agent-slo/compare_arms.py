#!/usr/bin/env python3
"""Compare two decode_only.py arms: token agreement first, then throughput.

The gate order matters. A throughput A/B is only admissible once the two arms
are known to compute the same thing, so this refuses to report a speedup
unless the greedy token streams agree, and it reports the agreement count
either way.

Usage:
    compare_arms.py <baseline.json> <candidate.json>
"""
from __future__ import annotations

import json
import statistics
import sys


def pct(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summary(blob):
    rows = blob["rows"]
    ntok = sum(r["ntok"] for r in rows)
    dur = blob["duration_s"]
    ttft = [r["ttft"] for r in rows if r["ttft"] is not None]
    # decode_only.py reports `tpot` as the request's total decode time in ms,
    # so dividing by the token count already yields ms/token.
    tpot = [r["tpot"] / r["ntok"] for r in rows if r["ntok"] and r["tpot"] is not None]
    return {
        "n": len(rows),
        "ntok": ntok,
        "tot_tps": ntok / dur,
        "tps_per_req": ntok / dur / len(rows),
        "ttft_p50": pct(ttft, 0.50),
        "ttft_p90": pct(ttft, 0.90),
        "ttft_p95": pct(ttft, 0.95),
        "ttft_p99": pct(ttft, 0.99),
        "tpot_p50_ms": statistics.median(tpot),
    }


def main():
    base = json.load(open(sys.argv[1]))
    cand = json.load(open(sys.argv[2]))

    b_rows = base["rows"]
    c_rows = cand["rows"]
    if len(b_rows) != len(c_rows):
        print(f"ROW COUNT DIFFERS: {len(b_rows)} vs {len(c_rows)} -- NOT COMPARABLE")
        return 2

    agree = 0
    len_agree = 0
    mismatches = []
    for i, (b, c) in enumerate(zip(b_rows, c_rows)):
        if b["ntok"] == c["ntok"]:
            len_agree += 1
        if b.get("text") == c.get("text"):
            agree += 1
        elif len(mismatches) < 5:
            mismatches.append((i, b["ntok"], c["ntok"], b.get("text", "")[:60], c.get("text", "")[:60]))

    n = len(b_rows)
    sb = summary(base)
    sc = summary(cand)

    # An empty stream is a failed request, not a fast one. Left in, it would
    # "agree" with the other arm's failure and let a broken cell report a clean
    # verdict, so this is checked before anything else is reported.
    empty = sum(1 for b, c in zip(b_rows, c_rows) if not b["ntok"] or not c["ntok"])
    if empty:
        print(f"EMPTY OUTPUT: {empty}/{n} pairs had an empty stream on one arm.")
        print("The cells did not serve the workload, so nothing here is a")
        print("measurement and no throughput number is admissible.")
        return 2

    print(f"agreement: {agree}/{n} token-for-token, {len_agree}/{n} same length")
    print()
    hdr = f"{'metric':<16}{'baseline':>14}{'fused':>14}{'ratio':>9}"
    print(hdr)
    print("-" * len(hdr))
    for key in ("total tokens", "tot_tps", "tps_per_req", "ttft_p50", "ttft_p90",
                "ttft_p95", "ttft_p99", "tpot_p50_ms"):
        k = "ntok" if key == "total tokens" else key
        bv, cv = sb[k], sc[k]
        ratio = cv / bv if bv else float("nan")
        print(f"{key:<16}{bv:>14.2f}{cv:>14.2f}{ratio:>9.3f}")

    if mismatches:
        print("\nfirst mismatches (idx, base_ntok, fused_ntok):")
        for i, bn, cn, bt, ct in mismatches:
            print(f"  [{i}] {bn} vs {cn}")
            print(f"      base : {bt!r}")
            print(f"      fused: {ct!r}")

    if agree != n:
        print("\nVERDICT: NOT ADMISSIBLE -- token streams differ, the two arms do")
        print("not compute the same thing, so no throughput comparison is valid.")
        return 1
    print("\nVERDICT: admissible -- token-for-token agreement at temperature 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
