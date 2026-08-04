#!/usr/bin/env python3
"""Which period is the limit cycle, before anything blocks by it?

ladder.py blocks consecutive window pairs, which is correct only if the cycle
has period 2. It does on vLLM. It does not on TokenSpeed serving a hybrid
model, where the observed series repeats every three windows:

    658.8, 633.9, 470.3, 655.2, 633.9, 474.0, 651.7, ...

Blocking a period-3 signal into pairs does not cancel the alternation, it
aliases it: the block means inherit a beat at the difference frequency, and the
SE computed over them is an SE of that beat rather than of the level. The same
run then reports a spread several times larger than it has, and a real effect
of a few percent stops being resolvable.

So: measure the period first, block by what is there.

Reports the autocorrelation at each candidate lag over the mean-removed series.
The dominant lag is the period. A series with no cycle shows no peak, and
period 1 (plain window mean) is then the right blocking.

Usage: period.py [SKIP_S] file.json [file.json ...]
"""

import json
import statistics as st
import sys

MAX_LAG = 8


def autocorr(xs, lag):
    """Pearson autocorrelation of `xs` at `lag`, or None if too short."""
    n = len(xs) - lag
    if n < 3:
        return None
    mean = st.mean(xs)
    dev = [x - mean for x in xs]
    num = sum(dev[i] * dev[i + lag] for i in range(n))
    den = sum(d * d for d in dev)
    return num / den if den else None


def blocked(xs, period):
    """Means of consecutive `period`-sized blocks, trailing remainder dropped."""
    return [
        st.mean(xs[i:i + period])
        for i in range(0, len(xs) - period + 1, period)
    ]


def analyse(path, skip_s):
    d = json.load(open(path))
    w = [x for x in d["windows"] if x["t_s"] > skip_s]
    # Prefer the server's own count; the client can undercount an engine that
    # packs several tokens into one SSE chunk.
    key = "server_tok_s" if any(x.get("server_tok_s") for x in w) else "output_tok_s"
    xs = [x[key] for x in w if x.get(key)]
    if len(xs) < 6:
        return {"file": path, "n": len(xs), "error": "too few windows"}

    acs = {lag: autocorr(xs, lag) for lag in range(1, MAX_LAG + 1)}
    acs = {k: v for k, v in acs.items() if v is not None}
    period = max(acs, key=lambda k: acs[k]) if acs else 1
    # A weak peak is no peak. Below this the series is not cycling and the
    # plain window mean is the honest block.
    if acs.get(period, 0) < 0.3:
        period = 1

    blocks = blocked(xs, period)
    mean = st.mean(blocks)
    se = st.stdev(blocks) / (len(blocks) ** 0.5) if len(blocks) > 1 else 0.0
    return {
        "file": path, "key": key, "n": len(xs), "period": period,
        "ac": acs, "mean": mean, "se": se, "blocks": len(blocks),
    }


def main():
    args = sys.argv[1:]
    skip = 0.0
    if args and not args[0].endswith(".json"):
        skip = float(args[0])
        args = args[1:]
    print(f"period detection over windows with t > {skip}s\n")
    print(f"{'file':<26}{'src':>14}{'n':>4}{'period':>8}{'ac':>8}"
          f"{'mean':>10}{'SE':>8}  autocorr by lag")
    for path in args:
        r = analyse(path, skip)
        if "error" in r:
            print(f"{path:<26}{r['error']:>30}")
            continue
        acs = " ".join(f"{lag}:{v:+.2f}" for lag, v in sorted(r["ac"].items()))
        name = r["file"].rsplit("/", 1)[-1].replace(".json", "")
        print(f"{name:<26}{r['key']:>14}{r['n']:>4}{r['period']:>8}"
              f"{r['ac'].get(r['period'], 0):>8.2f}{r['mean']:>10.1f}"
              f"{r['se']:>8.1f}  {acs}")


if __name__ == "__main__":
    main()
