#!/usr/bin/env python3
"""Prefill/decode round profile from a server log.

Usage: prefill_profile.py <server-log> [--window HH:MM:SS-HH:MM:SS]

Parses the scheduler's "Prefill batch." / "Decode batch." lines (RANK 0 only)
and reports, for the given window (default: the whole log):

  * prefill rounds, total seconds and share, plus the same for decode;
  * the split of prefill round duration by extend size and by cached prefix,
    which is where the cost of a long-context turn shows up;
  * decode round mean, batch size and generation throughput.

Round duration is the gap to the next round line. That is exact because rounds
are strictly serial: the scheduler logs one line per extend round and one per
`--decode-log-interval` decode rounds, and nothing runs beside them.
"""

import argparse
import datetime as dt
import re
import statistics as stat

LINE = re.compile(
    r"\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d),(\d{3})\s+ATTN TP RANK 0\] - "
    r"INFO - (Prefill|Mix|Decode) batch\. (.*)"
)
NEW = re.compile(r"#new-token: (\d+), #cached-token: (\d+)")
RUNNING = re.compile(r"#running-req: ([\d.]+)")
ACCEPT = re.compile(r"avg_accept_len: ([\d.]+)")
THROUGHPUT = re.compile(r"gen throughput \(token/s\): ([\d.]+)")
SEQ = re.compile(r"avg_seq_len: ([\d.]+)")
QUEUE = re.compile(r"#queue-req: (\d+)")

# Extend sizes worth separating: the chunk boundary is where the fast rounds sit.
SIZE_BANDS = [(0, 256), (256, 1024), (1024, 2048), (2048, 4000), (4000, 8193)]
CTX_BANDS = [(0, 50_000), (50_000, 100_000), (100_000, 10**9)]


def stamp(groups):
    day, hms, ms = groups
    return dt.datetime.combine(
        dt.date.fromisoformat(day),
        dt.time.fromisoformat(hms),
    ).replace(microsecond=int(ms) * 1000)


def parse(path):
    events = []
    with open(path, errors="ignore") as handle:
        for line in handle:
            match = LINE.search(line)
            if match:
                events.append((stamp(match.groups()[:3]), match.group(4), match.group(5)))
    events.sort()
    return events


def window_of(spec):
    start, _, end = spec.partition("-")
    today = dt.date.today()
    return (
        dt.datetime.combine(today, dt.time.fromisoformat(start)),
        dt.datetime.combine(today, dt.time.fromisoformat(end)),
    )


def mean(values):
    return stat.mean(values) if values else 0.0


def summarise(events, decode_interval):
    """Round accounting for a list of (when, kind, body) events.

    Returns a dict with the prefill/decode time shares, the prefill extend-size
    bands and the decode round stats, so callers can print or compare it without
    re-parsing.
    """
    prefill, decode = [], []
    for i, (when, kind, body) in enumerate(events):
        if i + 1 == len(events):
            break
        dur = (events[i + 1][0] - when).total_seconds()
        if kind == "Decode":
            decode.append((dur, parse_float(RUNNING, body), parse_float(ACCEPT, body),
                           parse_float(THROUGHPUT, body), parse_float(SEQ, body)))
            continue
        new = NEW.search(body)
        if not new:
            continue
        prefill.append((dur, int(new.group(1)), int(new.group(2)), kind,
                        parse_int(QUEUE, body), parse_float(RUNNING, body),
                        events[i + 1][1]))

    span = (events[-1][0] - events[0][0]).total_seconds()
    out = {
        "span": span,
        "start": events[0][0],
        "end": events[-1][0],
        "prefill": prefill,
        "decode": decode,
        "prefill_s": sum(p[0] for p in prefill),
        "decode_s": sum(d[0] for d in decode),
        "decode_rounds": len(decode) * decode_interval,
        "mixed": sum(1 for p in prefill if p[3] == "Mix"),
        "new_tokens": sum(p[1] for p in prefill),
        "cached_tokens": sum(p[2] for p in prefill),
        "bands": [],
        "decode_stats": None,
    }
    if prefill:
        for lo, hi in SIZE_BANDS:
            group = [p for p in prefill if lo <= p[1] < hi]
            # A prefill round followed by a decode line is charged only until the
            # next line, so its measured gap also holds that decode batch. Those
            # rounds are censored: the clean pairs (prefill followed by prefill)
            # are the ones that price the extend itself.
            clean = [p[0] for p in group if p[6] != "Decode"]
            out["bands"].append({
                "lo": lo, "hi": hi, "n": len(group),
                "mean_ms": mean([p[0] for p in group]) * 1000,
                "median_ms": (stat.median([p[0] for p in group]) * 1000) if group else 0.0,
                "total_s": sum(p[0] for p in group),
                "new_tokens": mean([p[1] for p in group]),
                "cached_tokens": mean([p[2] for p in group]),
                "n_clean": len(clean),
                "mean_clean_ms": mean(clean) * 1000,
            })
        out["queue_mean"] = mean([p[4] for p in prefill])
        out["running_mean"] = mean([p[5] for p in prefill])
    if decode:
        out["decode_stats"] = {
            "mean_ms": mean([d[0] for d in decode]) * 1000,
            "per_round_ms": out["decode_s"] / max(out["decode_rounds"], 1) * 1000,
            "running": mean([d[1] for d in decode]),
            "accept": mean([d[2] for d in decode]),
            "gen": mean([d[3] for d in decode]),
            "seq_len": mean([d[4] for d in decode]),
        }
    return out


def load_window(path, window=None, decode_interval=40):
    """Parse a log and return the summary for the whole file or a HH:MM:SS-HH:MM:SS window."""
    events = parse(path)
    if not events:
        return None
    if window:
        lo, hi = window
        day = events[0][0].date()
        lo = lo.replace(year=day.year, month=day.month, day=day.day)
        hi = hi.replace(year=day.year, month=day.month, day=day.day)
        events = [e for e in events if lo <= e[0] <= hi]
        if not events:
            return None
    return summarise(events, decode_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--window", help="HH:MM:SS-HH:MM:SS (times only, one day)")
    parser.add_argument("--decode-interval", type=int, default=40)
    args = parser.parse_args()

    stats = load_window(args.log, window_of(args.window) if args.window else None,
                        args.decode_interval)
    if stats is None:
        raise SystemExit(f"no scheduler batch lines in {args.log}")
    report(stats, args.log)


def report(stats, log):
    """Print the summary `summarise` returns."""
    prefill, decode = stats["prefill"], stats["decode"]
    span = stats["span"]
    prefill_s, decode_s = stats["prefill_s"], stats["decode_s"]
    print(f"log {log}")
    print(f"window {stats['start']:%H:%M:%S}-{stats['end']:%H:%M:%S}Z  span {span:.0f}s")
    print(f"prefill rounds {len(prefill)}  {prefill_s:.0f}s  {100 * prefill_s / span:.0f}% of span")
    print(f"decode rounds {stats['decode_rounds']} "
          f"({len(decode)} lines)  {decode_s:.0f}s  {100 * decode_s / span:.0f}%")
    print(f"sum {prefill_s + decode_s:.0f}s vs span {span:.0f}s "
          f"(gap {span - prefill_s - decode_s:.0f}s)")
    print(f"mixed rounds {stats['mixed']}")
    if prefill:
        print(f"new tokens {stats['new_tokens'] / 1e6:.2f}M  "
              f"cached tokens {stats['cached_tokens'] / 1e6:.2f}M  "
              f"{stats['new_tokens'] / max(prefill_s, 1e-9):.0f} new tok/s")
        print(f"queue-req mean {stats['queue_mean']:.1f}  "
              f"running-req mean {stats['running_mean']:.2f}")
        print("\nby extend size")
        print(f"  {'new-token band':>16} {'n':>5} {'mean':>8} {'median':>8} {'total':>7} {'newtok':>7} {'cached':>8}")
        for band in stats["bands"]:
            if not band["n"]:
                continue
            print(f"  {f'{band['lo']}-{band['hi']}':>16} {band['n']:>5} "
                  f"{band['mean_ms']:>7.0f}m {band['median_ms']:>7.0f}m "
                  f"{band['total_s']:>6.0f}s {band['new_tokens']:>7.0f} "
                  f"{band['cached_tokens']:>8.0f}")
        print("  (first mean is to the next log line; a round followed by a decode "
              "line therefore carries that decode batch)")
        print(f"  {'clean pairs':>16} {'n':>5} {'mean':>8}")
        for band in stats["bands"]:
            if not band["n_clean"]:
                continue
            print(f"  {f'{band['lo']}-{band['hi']}':>16} {band['n_clean']:>5} "
                  f"{band['mean_clean_ms']:>7.0f}m")
        print("\nby cached prefix, split at the chunk boundary (4096 tokens)")
        print(f"  {'cached band':>16} {'n<4096':>7} {'ms<4096':>8} {'n=4096':>7} {'ms=4096':>8}")
        for lo, hi in CTX_BANDS:
            small = [p[0] for p in prefill if lo <= p[2] < hi and p[1] < 4096]
            full = [p[0] for p in prefill if lo <= p[2] < hi and p[1] >= 4096]
            if not small and not full:
                continue
            print(f"  {f'{lo // 1000}k-{hi // 1000}k':>16} {len(small):>7} "
                  f"{mean(small) * 1000:>7.0f}m {len(full):>7} {mean(full) * 1000:>7.0f}m")
    if stats["decode_stats"]:
        d = stats["decode_stats"]
        print("\ndecode")
        print(f"  mean round {d['mean_ms']:.1f} ms  "
              f"per-round amortised {d['per_round_ms']:.1f} ms")
        print(f"  running-req mean {d['running']:.1f}  avg_seq_len {d['seq_len']:.0f}  "
              f"accept_len {d['accept']:.2f}  gen {d['gen']:.0f} tok/s")


def parse_float(pattern, body):
    match = pattern.search(body)
    return float(match.group(1)) if match else 0.0


def parse_int(pattern, body):
    match = pattern.search(body)
    return int(match.group(1)) if match else 0


if __name__ == "__main__":
    main()
