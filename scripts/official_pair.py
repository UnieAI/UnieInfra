#!/usr/bin/env python3
"""Side-by-side read-out of one official AgentX (3600 s) cell per engine.

`fair.py` answers "how fast are we relative to vLLM at the same interactivity".
This script answers the follow-up question that decides what to work on next:
*why* the ratio looks the way it does. For each concurrency it prints both
engines at the official window, how saturated the decode slots were, how much of
the prompt came out of the prefix cache, and how long requests queued before
their first token. A cell whose slots are already ~90% busy is throughput-bound
by step time and cannot be fixed by batching; a cell that is mostly idle is
queue-bound and can.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offtable import OFFICIAL_SECONDS, cells  # noqa: E402


def slot_stats(result_dir):
    """Decode-slot occupancy and prefix reuse for one result directory.

    Occupancy is sum(decode duration) / (slots * window): the fraction of the hour
    during which the engine was actually stepping decode with every slot busy. It
    is the difference between "make each step cheaper" and "feed the GPU better".
    """
    d = result_dir
    jl = os.path.join(d, "aiperf_artifacts", "profile_export.jsonl")
    summary = os.path.join(d, "aiperf_artifacts", "profile_export_aiperf.json")
    out = {}
    try:
        with open(summary) as fh:
            s = json.load(fh)
        out["prefix"] = s.get("overall_usage_prompt_cache_read_pct", {}).get("avg")
        out["ceiling"] = s.get("theoretical_prefix_cache_hit", {}).get("avg")
    except (OSError, ValueError):
        pass
    if not os.path.exists(jl):
        return out
    dec = 0.0
    n = 0
    with open(jl) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("metadata", {}).get("benchmark_phase") != "profiling":
                continue
            m = r.get("metrics", {}).get("full_decode_duration", {}).get("value")
            if isinstance(m, (int, float)):
                dec += m / 1000.0
                n += 1
    out["requests"] = n
    out["decode_s"] = dec
    return out


def slots_of(cell):
    """Decode slots for a cell. Ours is encoded in the tag (off16 = 16 slots);
    vLLM's is CONC/4 by construction in serve_vllm.sh, so it is inferred and
    returned with a flag that makes the printed number visibly approximate."""
    if cell.get("cap"):
        return cell["cap"], False
    if cell["engine"].startswith("vllm"):
        return max(1, cell["conc"] // 4), True
    return None, False


def main():
    cs = [c for c in cells() if c["dur"] == OFFICIAL_SECONDS and c.get("valid") and not c.get("noisy")]
    by = {}
    for c in cs:
        by.setdefault(c["conc"], {}).setdefault(c["engine"], []).append(c)
    if not by:
        print("no official cells yet")
        return
    print(f"{'conc':>5} {'engine':>10} {'cap':>4} {'RPM':>7} {'inter':>6} {'ITL90':>7} "
          f"{'TTFT90':>8} {'slots':>5} {'busy%':>6} {'prefix%':>8} {'ceil%':>6}")
    for conc in sorted(by):
        for eng in ("vllm", "unieinfra"):
            best = None
            for c in by[conc].get(eng, []):
                if best is None or c["rpm"] > best["rpm"]:
                    best = c
            if best is None:
                print(f"{conc:>5} {eng:>10} {'-':>4} {'-':>7} {'-':>6} {'-':>7} {'-':>8}")
                continue
            st = slot_stats(best["dir"])
            slots, inferred = slots_of(best)
            busy = ""
            if st.get("decode_s") and slots:
                busy = f"{st['decode_s'] / slots / OFFICIAL_SECONDS * 100:.0f}{'~' if inferred else ''}"
            prefix = f"{st['prefix']:.1f}" if st.get("prefix") else ""
            ceiling = f"{st['ceiling']:.1f}" if st.get("ceiling") else ""
            print(f"{conc:>5} {eng:>10} {str(best.get('cap')):>4} {best['rpm']:>7.1f} "
                  f"{best['inter']:>6.1f} {best['itl90']:>7.1f} {best['ttft90']:>8.0f} "
                  f"{str(slots):>5} {busy:>6} {prefix:>8} {ceiling:>6}")


if __name__ == "__main__":
    main()
