#!/usr/bin/env python3
"""Shared fair-comparison model for the UnieInfra-vs-vLLM sweep.

Fair means: same checkpoint, bfloat16, no quantization flag, 900 s official
AgentX window, no dropped requests. The MoE layout (tensor parallel or expert
parallel) is a serving configuration and is chosen per cell, so by default both
layouts are admitted and the frontier is built from the best cell per cap.
"""
import math
import sys

sys.path.insert(0, "/home/smc/workspace/roy/scripts")
from offtable import cells  # noqa: E402


OFFICIAL_SECONDS = 3600


def ref_duration():
    """The profile length both engines were measured at.

    A ratio only means something when the two sides ran the same window, so the
    preference is the documented recipe (the agentx preset's one hour, stored as
    dur=3600 for -dofficial cells) as soon as both engines have a valid cell
    there, and the 900 s screening window only until then.
    """
    sides = {}
    for c in cells():
        # Directories that miss the naming pattern carry no validity verdict.
        if c.get("valid") and not c.get("noisy"):
            sides.setdefault(c["dur"], set()).add(c["engine"] == "unieinfra")
    both = [d for d, s in sides.items() if len(s) == 2]
    if not both:
        return 900
    return OFFICIAL_SECONDS if OFFICIAL_SECONDS in both else max(both)


def reference():
    """vLLM reference point per concurrency: its fastest cell at ref_duration()."""
    dur = ref_duration()
    out = {}
    for c in cells():
        if not (c["engine"] or "").startswith("vllm") or c["dur"] != dur:
            continue
        if not c["valid"] or c["noisy"]:
            continue
        cur = out.get(c["conc"])
        if cur is None or c["rpm"] > cur["rpm"]:
            out[c["conc"]] = c
    return out


def variants(mode="any"):
    """Variant cells (mixed prefill/decode, non-default chunked prefill) and whether
    each has earned a place in the frontier.

    A variant is a scheduling experiment, not a serving configuration: it may only
    stand in for us once it beats the plain cell at the same (concurrency, cap) on
    both axes, so the comparison never improves by silently swapping in a different
    batch shape at a different interactivity.
    """
    pool = {}
    for c in eligible(mode):
        pool.setdefault(c["conc"], []).append(c)
    verdicts = []
    for conc, cs in pool.items():
        for c in cs:
            if not c.get("variant"):
                continue
            base = [o for o in cs if not o.get("variant") and o["cap"] == c["cap"]]
            ok = bool(base) and all(c["rpm"] > o["rpm"] and c["inter"] >= o["inter"] for o in base)
            verdicts.append({"cell": c, "beat": base, "accepted": ok})
    return sorted(verdicts, key=lambda v: (v["cell"]["conc"], v["cell"]["cap"] or 0))


def eligible(mode="any"):
    """UnieInfra cells that may be compared at all: at ref_duration(), capped,
    unquantized, valid, not noisy, and in the requested MoE layout."""
    dur = ref_duration()
    out = []
    for c in cells():
        if c["engine"] != "unieinfra" or c["dur"] != dur:
            continue
        if not c["cap"] or not c["valid"] or c["noisy"] or c["quant"]:
            continue
        if mode == "ep" and not c.get("ep"):
            continue
        if mode == "tp" and c.get("ep"):
            continue
        out.append(c)
    return out


def ours(mode="any"):
    """Admissible UnieInfra cells per concurrency, keyed by cap."""
    accepted = {id(v["cell"]) for v in variants(mode) if v["accepted"]}
    out = {}
    for c in eligible(mode):
        if c.get("variant") and id(c) not in accepted:
            continue
        # The same cap can exist in both MoE layouts, so the key carries the layout.
        out.setdefault(c["conc"], {})[(c["cap"], bool(c.get("ep")))] = c
    return {k: frontier(v.values()) for k, v in out.items()}


def frontier(cs):
    """Drop cells dominated at the same concurrency (no more RPM and no better
    per-user interactivity). Keeps the comparison honest: a slower, less
    interactive cell must never stand in for the config that wins."""
    cs = list(cs)
    keep = []
    for c in cs:
        dominated = any(
            o is not c
            and o["rpm"] >= c["rpm"]
            and o["inter"] >= c["inter"]
            and (o["rpm"] > c["rpm"] or o["inter"] > c["inter"])
            for o in cs
        )
        if not dominated:
            keep.append(c)
    return keep


def interp(pts, target):
    """RPM at `target` interactivity, interpolated in log-interactivity.

    Returns (rpm, status) where status is 'exact' or 'interp', else None with
    'need-smaller-cap' (all our cells are slower per user than the target, so
    interactivity has to be raised) or 'need-bigger-cap' (all are faster).
    """
    pts = sorted(pts, key=lambda c: c["inter"])
    for a in pts:
        if abs(a["inter"] - target) < 1e-9:
            return a["rpm"], "exact"
    for a, b in zip(pts, pts[1:]):
        if a["inter"] <= target <= b["inter"]:
            la, lb, lt = math.log(a["inter"]), math.log(b["inter"]), math.log(target)
            f = (lt - la) / (lb - la) if lb != la else 1.0
            return a["rpm"] + f * (b["rpm"] - a["rpm"]), "interp"
    if not pts:
        return None, "no-cells"
    # Fewer sequences per step means faster per user: so when even our most
    # interactive cell is slower than the target, the missing cell has a smaller cap.
    return None, "need-smaller-cap" if pts[-1]["inter"] < target else "need-bigger-cap"


TOL = 0.25  # relative interactivity tolerance for calling two points "matched"
TOL_WIDE = 0.5  # looser band, reported as "near-wide" so it is never silent
TOL_TIGHT = 0.02  # a nearest point this close is reported as "tight", not "near":
#                   a cap cannot always be chosen finely enough to land exactly on
#                   the reference interactivity (c4 has at most 4 useful slots).


def matched(mode="any", tol=TOL):
    """One row per concurrency: our RPM compared at the vLLM interactivity.

    A concurrency counts as matched when our cells bracket the reference
    interactivity (interpolated in log-interactivity), or when the nearest cell
    is within `tol` of it (used directly, status "near"). Otherwise the row says
    which direction the missing cell has to go.
    """
    ref, mine = reference(), ours(mode)
    rows = []
    for conc in sorted(ref):
        r = ref[conc]
        pts = mine.get(conc, [])
        rpm, status = interp(pts, r["inter"])
        if rpm is None and pts:
            near = min(pts, key=lambda c: abs(math.log(c["inter"] / r["inter"])))
            gap = abs(math.log(near["inter"] / r["inter"]))
            if gap <= math.log(1 + TOL_TIGHT):
                rpm, status = near["rpm"], "tight"
            elif gap <= math.log(1 + tol):
                rpm, status = near["rpm"], "near"
            elif gap <= math.log(1 + TOL_WIDE):
                rpm, status = near["rpm"], "near-wide"
        rows.append(
            {
                "conc": conc,
                "ref_rpm": r["rpm"],
                "ref_inter": r["inter"],
                "ref_ttft90": r["ttft90"],
                "ref_outs": r["outs"],
                "rpm": rpm,
                "ratio": (rpm / r["rpm"]) if rpm else None,
                "status": status,
                "cells": pts,
            }
        )
    return rows


def geomean(rows):
    rs = [r["ratio"] for r in rows if r["ratio"]]
    if not rs:
        return None
    return math.exp(sum(math.log(x) for x in rs) / len(rs))


def floor(row):
    """Lower-bound ratio for a concurrency the reference cannot reach.

    When the reference's own cells never get as interactive as our least
    interactive cell, there is no matched point to compare at. The honest
    statement is then a lower bound: our biggest-cap (highest-throughput) cell
    already beats the reference's best cell while still being faster per user,
    so any truly matched comparison would flatter us more, not less.
    """
    if row["ratio"] is not None or not row["cells"]:
        return None
    pts = sorted(row["cells"], key=lambda c: c["cap"])
    top = pts[-1]
    if top["inter"] <= row["ref_inter"] or not top["rpm"]:
        return None
    return {
        "conc": row["conc"],
        "our_rpm": top["rpm"],
        "our_inter": top["inter"],
        "ref_rpm": row["ref_rpm"],
        "ref_inter": row["ref_inter"],
        "ratio": top["rpm"] / row["ref_rpm"],
    }


def fmt(row):
    if row["ratio"] is None:
        return f"c{row['conc']:<3d} {row['status']:<11s} (need interactivity {row['ref_inter']:.0f})"
    return (
        f"c{row['conc']:<3d} {row['rpm']:6.1f} vs {row['ref_rpm']:6.1f} RPM @ {row['ref_inter']:5.0f}"
        f" -> {row['ratio']:4.2f}x {row['status']}"
    )


if __name__ == "__main__":
    mode = "ep" if "--ep" in sys.argv else ("tp" if "--tp" in sys.argv else "any")
    print(f"profile {ref_duration()}s per cell ({'agentx preset' if ref_duration() == OFFICIAL_SECONDS else 'screening window'})")
    rows = matched(mode)
    for r in rows:
        print(fmt(r))
    g = geomean(rows)
    n = sum(1 for r in rows if r["ratio"])
    print(f"\nmode={mode}  matched {n}/{len(rows)}  geomean {g:.3f}x" if g else f"mode={mode}  no matched points")
    bounds = [b for b in (floor(r) for r in rows) if b]
    for b in bounds:
        print(
            f"c{b['conc']:<3d} lower bound {b['ratio']:4.2f}x "
            f"(our cap-ceiling cell {b['our_rpm']:.1f} RPM @ {b['our_inter']:.0f} vs reference "
            f"{b['ref_rpm']:.1f} @ {b['ref_inter']:.0f}; the reference never gets this interactive)"
        )
    if bounds and g:
        allr = [r["ratio"] for r in rows if r["ratio"]] + [b["ratio"] for b in bounds]
        gg = math.exp(sum(math.log(x) for x in allr) / len(allr))
        print(f"geomean including the lower bound(s) {gg:.3f}x over {len(allr)} rows (not a matched comparison)")
    for v in variants(mode):
        c, base = v["cell"], v["beat"]
        ref = ", ".join(f"cap{o['cap']} {o['rpm']:.1f}@{o['inter']:.0f}" for o in base) or "no plain cell"
        tag = "accepted" if v["accepted"] else "excluded"
        print(
            f"variant c{c['conc']:<3d} cap{c['cap']} {c['variant']} "
            f"{c['rpm']:.1f} RPM @ {c['inter']:.0f} vs {ref} -> {tag}"
        )
