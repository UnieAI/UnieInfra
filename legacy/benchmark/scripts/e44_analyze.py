#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""E44 step 1 analysis: segment a vLLM chrome trace into decode steps and
attribute GPU time by kernel category.

Reads the chrome trace produced by `e44_vllm_decode_probe.py`, finds the
steady-state decode region, and reports per step:
  * wall time, GPU-busy time (union of device intervals), idle time & gap
    histogram,
  * kernel launch count,
  * time share per category and per individual kernel.

Segmentation is data-driven: a "marker" kernel that fires exactly once per
forward pass delimits steps; the decode region is the tail run of steps whose
duration is within tolerance of the median tail step.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

DEVICE_CATS = {"kernel", "gpu_memset", "gpu_memcpy"}

CATEGORY_RULES = [
    (
        "attention",
        re.compile(
            r"flash_fwd|flash::|paged_attention|merge_attn_states|"
            r"attn_fwd|_attention_kernel|flashinfer.*(prefill|decode)"
        ),
    ),
    ("kv_cache_write", re.compile(r"reshape_and_cache|cache_kernel|copy_blocks")),
    (
        "weight_gemm",
        re.compile(
            r"gemm|gemv|cublas|cutlass|splitKreduce|s16816|" r"tensorop|scaled_mm|nvjet"
        ),
    ),
    ("rope", re.compile(r"rotary|rope")),
    (
        "norm_act",
        re.compile(
            r"rms_norm|rmsnorm|layernorm|silu|swiglu|gelu|"
            r"triton_red_fused|triton_poi_fused"
        ),
    ),
    (
        "sampling",
        re.compile(
            r"sampl|top_k|top_p|topk|topp|softmax|argmax|multinomial|"
            r"renorm|gumbel|categorical|radix|sort|cub::|Reduce|"
            r"random|philox"
        ),
    ),
    (
        "copy_glue",
        re.compile(
            r"elementwise_kernel|direct_copy|vectorized_|"
            r"CatArrayBatched|index_select|gather|scatter|"
            r"unrolled_elementwise|fill_|memset|Memcpy|memcpy|"
            r"copy_device_to_device"
        ),
    ),
]


def categorize(name: str) -> str:
    for cat, rx in CATEGORY_RULES:
        if rx.search(name):
            return cat
    return "other"


def union_busy(intervals):
    """Total wall time covered by the union of (start, end) intervals."""
    if not intervals:
        return 0.0, []
    iv = sorted(intervals)
    busy = 0.0
    gaps = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            gaps.append(s - ce)
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    return busy, gaps


def short(name: str) -> str:
    """Human-readable kernel label."""
    n = name
    m = re.search(r"cutlass_\w+?gemm\w*?_(\d+x\d+_\d+x\d+)", n)
    if "cutlass" in n and m:
        base = re.search(r"(cutlass_\d+_tensorop_\w+?gemm)", n)
        return f"{base.group(1) if base else 'cutlass_gemm'}_{m.group(1)}"
    for pat in (
        r"(triton_\w+)",
        r"vllm::(\w+)",
        r"flash::(flash_fwd_\w+)",
        r"cublasLt::(\w+)",
        r"(nvjet\w*)",
    ):
        mm = re.search(pat, n)
        if mm:
            return mm.group(1)
    mm = re.search(r"at::native::(\w+)", n)
    if mm:
        tail = re.findall(r"lambda\((\w[\w:]*)\)", n)
        return f"{mm.group(1)}<{tail[0] if tail else '?'}>"
    return n[:70]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--marker",
        default=None,
        help="kernel substring that fires once per forward pass",
    )
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    d = json.load(open(args.trace))
    ev = d["traceEvents"] if isinstance(d, dict) else d
    dev = [e for e in ev if e.get("cat") in DEVICE_CATS and e.get("dur") is not None]
    dev.sort(key=lambda e: e["ts"])
    print(f"device events: {len(dev)}")

    # ---- find a marker kernel: one launch per forward pass ------------------
    cnt = Counter(e["name"] for e in dev)
    # reshape_and_cache fires once per layer per pass; a per-pass marker is one
    # whose count is small and evenly spaced. Prefer explicit --marker.
    marker = args.marker
    if marker is None:
        # Use the LAST kernel of each pass instead: sampling touches the logits
        # tensor once per pass. Fall back to the largest-N kernel that is a
        # multiple of nothing else -- instead we segment on time gaps.
        marker = None

    ivs = [(e["ts"], e["ts"] + e["dur"]) for e in dev]
    whole_busy, all_gaps = union_busy(ivs)
    whole_wall = max(e[1] for e in ivs) - min(e[0] for e in ivs)
    print(
        f"whole capture: wall={whole_wall/1000:.1f}ms busy={whole_busy/1000:.1f}ms "
        f"({100*whole_busy/whole_wall:.2f}%)"
    )
    if all_gaps:
        gs = sorted(all_gaps, reverse=True)
        print(f"largest inter-kernel gaps (us): {[round(g,1) for g in gs[:12]]}")

    # ---- segment on a marker kernel that fires once per forward pass -------
    if marker is None:
        # A per-pass marker appears exactly N times where N is the number of
        # forward passes; pick the most common such count among low-count
        # kernels and take a marker with that count.
        low = {n: c for n, c in cnt.items() if c <= 200}
        if not low:
            raise SystemExit("no marker candidate; pass --marker")
        npass = Counter(low.values()).most_common(1)[0][0]
        cands = [n for n, c in low.items() if c == npass]
        # prefer a sampling-ish marker (end of step) for stable boundaries
        pref = [n for n in cands if "sample" in n or "post_update" in n]
        marker = (pref or cands)[0]
    print(f"marker kernel ({cnt.get(marker, 0)}x): {marker[:100]}")

    steps = []
    cur = []
    for e in dev:
        cur.append(e)
        if e["name"] == marker:
            steps.append(cur)
            cur = []
    if cur:
        steps.append(cur)
    print(f"segments found: {len(steps)}")

    def span(st):
        s = min(e["ts"] for e in st)
        en = max(e["ts"] + e["dur"] for e in st)
        return s, en

    info = []
    for i, st in enumerate(steps):
        s, en = span(st)
        busy, gaps = union_busy([(e["ts"], e["ts"] + e["dur"]) for e in st])
        info.append(
            {
                "i": i,
                "n_kernels": len(st),
                "wall_us": en - s,
                "busy_us": busy,
                "gaps": gaps,
            }
        )
    for it in info:
        print(
            f"  seg{it['i']:3d} n={it['n_kernels']:6d} wall={it['wall_us']/1000:8.3f}ms "
            f"busy={it['busy_us']/1000:8.3f}ms ({100*it['busy_us']/it['wall_us']:.1f}%)"
        )

    # ---- pick the steady-state decode region -------------------------------
    # Decode steps all have the same kernel count; prefill steps differ.
    kc = Counter(it["n_kernels"] for it in info)
    decode_n = kc.most_common(1)[0][0]
    dec = [it for it in info if it["n_kernels"] == decode_n]
    # drop the first (may still be warming) if we have plenty
    if len(dec) > 4:
        dec = dec[1:]
    print(f"\ndecode steps: {len(dec)} segments with {decode_n} kernels each")

    walls = sorted(it["wall_us"] for it in dec)
    busies = sorted(it["busy_us"] for it in dec)
    med_wall = walls[len(walls) // 2]
    med_busy = busies[len(busies) // 2]
    all_gaps = [g for it in dec for g in it["gaps"]]
    all_gaps.sort()

    print(f"median step wall  : {med_wall/1000:.3f} ms")
    print(
        f"median step busy  : {med_busy/1000:.3f} ms  "
        f"({100*med_busy/med_wall:.2f}% GPU-busy)"
    )
    print(f"kernels per step  : {decode_n}")
    if all_gaps:
        import statistics

        print(
            f"intra-step gaps   : n={len(all_gaps)/len(dec):.0f}/step "
            f"mean={statistics.mean(all_gaps):.3f}us "
            f"median={statistics.median(all_gaps):.3f}us "
            f"max={max(all_gaps):.3f}us "
            f"total={sum(all_gaps)/len(dec):.1f}us/step "
            f"({100*sum(all_gaps)/len(dec)/med_wall:.2f}% of step)"
        )
        over1 = [g for g in all_gaps if g > 1.0]
        print(
            f"gaps > 1us        : {len(over1)/len(dec):.1f}/step, "
            f"{sum(over1)/len(dec):.1f}us/step"
        )

    # ---- per-category and per-kernel attribution over decode steps ---------
    dec_idx = {it["i"] for it in dec}
    cat_t = defaultdict(float)
    cat_n = defaultdict(int)
    ker_t = defaultdict(float)
    ker_n = defaultdict(int)
    ker_cat = {}
    for i, st in enumerate(steps):
        if i not in dec_idx:
            continue
        for e in st:
            c = categorize(e["name"])
            cat_t[c] += e["dur"]
            cat_n[c] += 1
            k = short(e["name"])
            ker_t[k] += e["dur"]
            ker_n[k] += 1
            ker_cat[k] = c

    nsteps = len(dec)
    tot = sum(cat_t.values())
    print(
        f"\ntotal kernel time per step: {tot/nsteps/1000:.3f} ms "
        f"(sum-of-durations; busy={med_busy/1000:.3f} ms)"
    )
    print(f"\n{'category':<16} {'us/step':>10} {'% of step':>10} {'n/step':>8}")
    rows = []
    for c, t in sorted(cat_t.items(), key=lambda x: -x[1]):
        us = t / nsteps
        rows.append(
            {
                "category": c,
                "us_per_step": us,
                "pct_of_step": 100 * us / med_wall,
                "n_per_step": cat_n[c] / nsteps,
            }
        )
        print(f"{c:<16} {us:>10.1f} {100*us/med_wall:>9.2f}% {cat_n[c]/nsteps:>8.1f}")

    print(f"\n{'kernel':<62} {'cat':<15} {'us/step':>9} {'%step':>7} {'n':>5}")
    krows = []
    for k, t in sorted(ker_t.items(), key=lambda x: -x[1])[: args.top]:
        us = t / nsteps
        krows.append(
            {
                "kernel": k,
                "category": ker_cat[k],
                "us_per_step": us,
                "pct_of_step": 100 * us / med_wall,
                "n_per_step": ker_n[k] / nsteps,
            }
        )
        print(
            f"{k:<62} {ker_cat[k]:<15} {us:>9.2f} {100*us/med_wall:>6.2f}% "
            f"{ker_n[k]/nsteps:>5.1f}"
        )

    if args.out:
        json.dump(
            {
                "trace": args.trace,
                "decode_steps": nsteps,
                "kernels_per_step": decode_n,
                "step_wall_us": med_wall,
                "step_busy_us": med_busy,
                "gpu_busy_pct": 100 * med_busy / med_wall,
                "gap_total_us_per_step": sum(all_gaps) / nsteps if all_gaps else 0,
                "categories": rows,
                "kernels": krows,
            },
            open(args.out, "w"),
            indent=1,
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
