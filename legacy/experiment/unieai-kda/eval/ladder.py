#!/usr/bin/env python3
"""Rung-by-rung comparison using period-blocked means.

Why not plateau.py's window mean: on Qwen3-32B the throughput settles into a
period-2 limit cycle (prefill burst, then decode burst) with swings up to 50%
of the mean. Averaging raw windows inherits that spread -- L2 read as
547.3 +/- 203 -- which made a real 4.75% effect look unmeasurable.

Consecutive windows pair into one full period. Those pair means are nearly
free of the alternation: the same L2 run gives 547.3 +/- 0.4. Same data, same
estimate of the level, ~500x tighter uncertainty, because the variance being
removed was structure rather than noise.

Reports SE over periods, so a difference between rungs can be stated in sigma
instead of eyeballed. Autocorrelation between adjacent periods would still bias
SE low; the periods observed so far look independent (L2: 548 546 546 548 548
548), but treat single-digit sigma with more suspicion than the number suggests.

--drop N discards the first N periods of the FIRST file only, for runs whose
opening period contains a startup stall.

Usage: ladder.py [--drop N] [SKIP_S] a.json b.json ...
"""

import json
import statistics as st
import sys


def periods(path, skip_s, drop=0):
    d = json.load(open(path))
    # Use the SERVER's own token counter, not the client's.
    #
    # The client counts SSE chunks. Under speculative decoding one chunk carries
    # every token accepted in that step, so the client undercounts by the
    # acceptance factor -- client_captures_frac ran 0.34-0.69 on every
    # speculative arm while baselines sat at 1.00. Reading output_tok_s turned
    # a +45% MTP gain into a reported -42% loss, and the discrepancy was sitting
    # in the same JSON the whole time. server_tok_s comes from
    # vllm:generation_tokens_total and is not subject to that.
    key = "server_tok_s" if any(
        w.get("server_tok_s") for w in d["windows"]) else "output_tok_s"
    v = [w[key] for w in d["windows"] if w["t_s"] > skip_s and w.get(key)]
    p = [(v[i] + v[i + 1]) / 2 for i in range(0, len(v) - 1, 2)][drop:]
    if not p:
        return None
    se = st.pstdev(p) / (len(p) ** 0.5) if len(p) > 1 else float("nan")
    return dict(mean=st.mean(p), se=se, n=len(p), vals=p, src=key,
                cfrac=d.get("client_captures_frac"),
                failed=d.get("total_failed", 0))


def main(argv):
    drop = 0
    if argv and argv[0] == "--drop":
        drop = int(argv[1])
        argv = argv[2:]
    skip = 60.0
    if argv and not argv[0].endswith(".json"):
        skip = float(argv[0])
        argv = argv[1:]

    rows = []
    for i, f in enumerate(argv):
        r = periods(f, skip, drop if i == 0 else 0)
        if r is None:
            print(f"{f}: no periods")
            continue
        rows.append((f.rsplit("/", 1)[-1].replace(".json", ""), r))

    if not rows:
        return
    base = rows[0][1]["mean"]
    print(f"period-blocked, windows t > {skip}s\n")
    print(f"{'rung':<22}{'tok/s':>9}{'SE':>7}{'n':>4}{'vs prev':>12}{'vs first':>11}  periods")
    prev = None
    for name, r in rows:
        if prev is None:
            vs_prev = "--"
        else:
            d = r["mean"] - prev["mean"]
            se = (r["se"] ** 2 + prev["se"] ** 2) ** 0.5
            sig = abs(d) / se if se else float("inf")
            vs_prev = f"{d / prev['mean']:+.2%} ({sig:.0f}s)"
        flag = f"  FAILED={r['failed']}" if r["failed"] else ""
        # Surface the client/server mismatch instead of leaving it in the file.
        # It was there, correct, and ignored -- being computed is not the same
        # as being seen.
        cf = r.get("cfrac")
        if cf is not None and abs(cf - 1.0) > 0.02:
            flag += f"  [client counted {cf:.0%} of server tokens]"
        if r.get("src") != "server_tok_s":
            flag += "  [no server counter, client numbers unverified]"
        print(f"{name:<22}{r['mean']:>9.1f}{r['se']:>7.1f}{r['n']:>4}{vs_prev:>12}"
              f"{(r['mean'] - base) / base:>+11.2%}  {[round(x) for x in r['vals']]}{flag}")
        prev = r


if __name__ == "__main__":
    main(sys.argv[1:])
