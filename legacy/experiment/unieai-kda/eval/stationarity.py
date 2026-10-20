#!/usr/bin/env python3
"""Is a run's plateau actually a plateau?

plateau.py assumes windows past SKIP_S are steady and averages them. On
Qwen3-32B at C=64 that assumption broke silently: L0's windows were still
diverging at t=240 (lows 455->402, highs 622->714) while L1's were converging
(lows 412->482, highs 666->568). Two non-stationary series, caught in different
phases of a transient -- their means differed by 2%, which meant nothing.

A mean is only worth reporting if the series it summarises has settled. This
checks that before the comparison, not after a suspicious result.

Three signals, because each misses a different failure:

  drift    second-half mean vs first-half mean. Catches monotone trends. Blind
           to oscillation, which has equal halves by construction.
  swing    peak-to-peak of the alternation, as a fraction of the mean. Large
           swing means the mean is an average over a limit cycle rather than a
           level -- still usable, but only if enough full periods are covered.
  envelope whether the alternation's highs and lows are themselves converging
           or diverging. This is the one that caught the Qwen3-32B problem:
           swing was large in both arms and drift was small in both, but the
           envelopes were moving in opposite directions.

Usage: stationarity.py [SKIP_S] file.json [file.json ...]
"""

import json
import statistics as st
import sys


def analyse(path, skip_s):
    d = json.load(open(path))
    w = [x for x in d["windows"] if x["t_s"] > skip_s]
    if len(w) < 6:
        return None
    v = [x["output_tok_s"] for x in w]
    n = len(v)
    half = n // 2
    first, second = st.mean(v[:half]), st.mean(v[half:])
    mean = st.mean(v)

    # Split into the two alternating phases; on a period-2 limit cycle these
    # separate cleanly into "lows" and "highs".
    even, odd = v[0::2], v[1::2]
    lo, hi = (even, odd) if st.mean(even) < st.mean(odd) else (odd, even)

    def trend(seq):
        """Signed fractional change from the first third to the last third."""
        k = max(1, len(seq) // 3)
        a, b = st.mean(seq[:k]), st.mean(seq[-k:])
        return (b - a) / a if a else 0.0

    lo_t, hi_t = trend(lo), trend(hi)
    # Envelope diverges when lows fall while highs rise (or vice versa).
    envelope = hi_t - lo_t

    return dict(
        n=n,
        mean=mean,
        drift=(second - first) / first if first else 0.0,
        swing=(st.mean(hi) - st.mean(lo)) / mean if mean else 0.0,
        lo_trend=lo_t,
        hi_trend=hi_t,
        envelope=envelope,
        failed=d.get("total_failed", 0),
    )


def main(argv):
    skip = 60.0
    files = argv
    if argv and not argv[0].endswith(".json"):
        skip = float(argv[0])
        files = argv[1:]

    print(f"stationarity over windows with t > {skip}s\n")
    print(f"{'file':<24}{'n':>4}{'mean':>9}{'drift':>9}{'swing':>9}"
          f"{'lo_trend':>10}{'hi_trend':>10}{'envelope':>10}  verdict")
    for f in files:
        r = analyse(f, skip)
        name = f.rsplit("/", 1)[-1].replace(".json", "")
        if r is None:
            print(f"{name:<24}  too few windows")
            continue
        # Thresholds are judgement calls, stated so they can be argued with:
        # 5% drift or 8% envelope movement is enough to make a 2-5% effect
        # unmeasurable, which is the size of effect this ladder is chasing.
        bad = []
        if abs(r["drift"]) > 0.05:
            bad.append("DRIFTING")
        # The envelope check only means something when there is an envelope.
        # Below a few percent swing the "lows" and "highs" series are the same
        # distribution split arbitrarily, so their trends are noise and their
        # difference is noise doubled -- ld_q32_L3 came back at -80.6% envelope
        # on a 3.9% swing and 0.0% drift, which is a settled run, not a moving
        # one. Require a real limit cycle before judging whether it is stable.
        elif r["swing"] > 0.10 and abs(r["envelope"]) > 0.08:
            bad.append("ENVELOPE-MOVING")
        if r["failed"]:
            bad.append(f"FAILED={r['failed']}")
        verdict = " ".join(bad) if bad else "settled"
        print(f"{name:<24}{r['n']:>4}{r['mean']:>9.1f}{r['drift']:>+8.1%}"
              f"{r['swing']:>+8.1%}{r['lo_trend']:>+9.1%}{r['hi_trend']:>+9.1%}"
              f"{r['envelope']:>+9.1%}  {verdict}")


if __name__ == "__main__":
    main(sys.argv[1:])
