#!/usr/bin/env python3
"""Plateau mean over a run's windows, discarding ramp-up.

Replaces the convergence test for signals that oscillate instead of settling.
The split arm swings ~3% with a ~6-window period, so any "N consecutive windows
within X%" rule either never fires or fires on a peak; averaging the whole
plateau is both robust to that and reports the swing honestly as sd/range.

Usage: plateau.py [SKIP_S] file.json [file.json ...]
"""

import json
import os
import statistics as st
import sys


def plateau(path, skip_s):
    d = json.load(open(path))
    w = [x for x in d["windows"] if x["t_s"] > skip_s]
    if not w:
        return None
    v = [x["output_tok_s"] for x in w]
    srv = [x["server_tok_s"] for x in w if x.get("server_tok_s")]
    return dict(
        n=len(v), mean=st.mean(v),
        sd=st.stdev(v) if len(v) > 1 else 0.0,
        lo=min(v), hi=max(v),
        srv=st.mean(srv) if srv else float("nan"),
        mism=d.get("total_token_mismatch"), fail=d.get("total_failed"),
    )


def main(argv):
    skip_s = 60.0
    if argv and argv[0].replace(".", "").isdigit():
        skip_s = float(argv[0]); argv = argv[1:]
    print(f"plateau = windows with t > {skip_s:.0f}s\n")
    print(f"{'file':26s} {'win':>4s} {'mean':>8s} {'sd':>6s} "
          f"{'min':>8s} {'max':>8s} {'server':>8s} {'fail/mm':>9s}")
    rows = []
    for f in argv:
        base = os.path.basename(f).replace(".json", "")
        r = plateau(f, skip_s)
        if not r:
            print(f"{base:26s}  (no windows past skip)")
            continue
        rows.append((base, r))
        print(f"{base:26s} {r['n']:4d} {r['mean']:8.1f} {r['sd']:6.1f} "
              f"{r['lo']:8.1f} {r['hi']:8.1f} {r['srv']:8.1f} "
              f"{str(r['fail'])+'/'+str(r['mism']):>9s}")
    on = [r["mean"] for b, r in rows if "_on_" in b]
    off = [r["mean"] for b, r in rows if "_off_" in b]
    if on and off:
        print(f"\nON  n={len(on)} mean={st.mean(on):.1f}  {[round(x,1) for x in on]}")
        print(f"OFF n={len(off)} mean={st.mean(off):.1f}  {[round(x,1) for x in off]}")
        print(f"split effect: {100*(st.mean(on)/st.mean(off)-1):+.2f}%")
        if len(on) > 1 and len(off) > 1:
            print(f"worst ON {min(on):.1f} vs best OFF {max(off):.1f} -> "
                  f"overlap={min(on) < max(off)}")


if __name__ == "__main__":
    main(sys.argv[1:])
