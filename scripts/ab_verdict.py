#!/usr/bin/env python3
"""One command verdict for an A/B cell against the production baseline.

Usage: ab_verdict.py <variant-tag> [--base off16] [--conc 64]

Finds the newest cell of each tag, prints the standard row (RPM, interactivity,
ITL p90, TTFT p90), then the scheduler-round profile of both cells over their
profiling windows: prefill/decode split, the per-extend-size round table and the
decode round cost. That is what tells a scheduling or serve-flag A/B apart from
noise, and what makes the next experiment obvious.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offtable
import prefill_profile


def newest(tag, conc):
    found = [
        c for c in offtable.cells()
        if c["tag"] == tag and c["conc"] == conc and c["valid"] and not c["noisy"]
    ]
    return max(found, key=lambda c: c["stamp"]) if found else None


def server_log(cell):
    """The server log whose start precedes the cell's agentx start."""
    logs = sorted(glob.glob(f"/home/smc/workspace/roy/logs/unieinfra-server-c{cell['conc']}-*.log"))
    candidates = [p for p in logs if os.path.basename(p).split("-")[-1][:-4] <= cell["stamp"]]
    return candidates[-1] if candidates else None


def profile(cell):
    """Round profile over the profiling window: the last `dur` seconds of the log."""
    log = server_log(cell)
    if not log:
        return None, None
    events = prefill_profile.parse(log)
    if not events:
        return log, None
    end = events[-1][0]
    start = end.timestamp() - cell["dur"]
    window = [e for e in events if e[0].timestamp() >= start]
    if len(window) < 10:
        return log, None
    # declog cells set --decode-log-interval 1: every decode round has its own
    # line, so nothing is amortised and every gap is one round's own cost.
    interval = 1 if cell.get("variant") == "declog" else 40
    return log, prefill_profile.summarise(window, interval)


def row(cell):
    return (f"c{cell['conc']} {cell['tag']:<12} {cell['dur']:>4}s "
            f"RPM {cell['rpm']:>6.1f}  inter {cell['inter']:>6.1f}  "
            f"ITL90 {cell['itl90']:>6.1f}ms  TTFT90 {cell['ttft90']:>6.1f}s  "
            f"out {cell['outs'] or 0:>6.1f} tok/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--base", default="off16")
    parser.add_argument("--conc", type=int, default=64)
    args = parser.parse_args()

    base = newest(args.base, args.conc)
    variant = newest(args.tag, args.conc)
    if base is None:
        raise SystemExit(f"no valid cell for baseline tag {args.base} at c{args.conc}")
    if variant is None:
        raise SystemExit(f"no valid cell for tag {args.tag} at c{args.conc}")

    print(row(base))
    print(row(variant))
    if base["rpm"]:
        print(f"RPM ratio {variant['rpm'] / base['rpm']:.2f}x   "
              f"interactivity ratio {variant['inter'] / base['inter']:.2f}x   "
              f"TTFT90 ratio {variant['ttft90'] / base['ttft90']:.2f}x")
    print()

    for label, cell in (("baseline", base), ("variant", variant)):
        log, stats = profile(cell)
        if stats is None:
            print(f"{label}: no scheduler rounds found ({log})")
            continue
        prefill, decode = stats["prefill"], stats["decode"]
        print(f"{label} {cell['tag']} — {os.path.basename(log or '')}")
        print(f"  prefill {len(prefill)} rounds {stats['prefill_s']:.0f}s "
              f"({100 * stats['prefill_s'] / stats['span']:.0f}%), "
              f"decode {stats['decode_rounds']} rounds {stats['decode_s']:.0f}s "
              f"({100 * stats['decode_s'] / stats['span']:.0f}%)")
        if decode:
            d = stats["decode_stats"]
            print(f"  decode step {d['per_round_ms']:.1f} ms amortised, "
                  f"batch {d['running']:.1f}, gen {d['gen']:.0f} tok/s")
        if prefill:
            print(f"  prefill rounds by extend size (#new-token), ms mean / total s / count")
            cells = " | ".join(
                f"{b['lo']}-{b['hi']}: {b['mean_ms']:.0f} / {b['total_s']:.0f}s / {b['n']}"
                for b in stats["bands"] if b["n"]
            )
            print(f"    {cells}")
        print()


if __name__ == "__main__":
    main()
