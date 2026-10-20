#!/usr/bin/env python3
"""Decompose a decode step from a torch-profiler chrome trace.

Same construction as E36:
  * steps are delimited by a kernel that fires exactly once per decode step
    (`fused_decode_input_prep_kernel`);
  * GPU-busy is the *union* of device-event intervals (kernel u memcpy u
    memset), so overlapping streams are never double counted;
  * everything is reported per step, over the interior steps only.

Usage: decomp.py <trace.json[.gz]> [--label X]
"""
import gzip
import json
import re
import sys
from collections import defaultdict

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}

# Kernel-name -> bucket. First match wins; order matters.
RULES = [
    ("gemm",       re.compile(r"cutlass.*gemm|splitKreduce|gemv|_gemm|nvjet|"
                              r"^ampere_|^sm\d+_xmma|cublas|GemmUniversal|"
                              r"tensorop.*gemm", re.I)),
    ("attn_full",  re.compile(r"_fwd_grouped_kernel_stage|_fwd_kernel_stage|"
                              r"flash_fwd|_decode_grouped|paged_attention|"
                              r"merge_state|BatchDecodeWithPagedKV|"
                              r"mha_decode|attn_merge", re.I)),
    ("attn_gdn",   re.compile(r"delta_rule|gated_delta|recurrent|chunk_h|chunk_o|"
                              r"gdn_|fused_recurrent|_state_passing|causal_conv|"
                              r"conv1d|short_conv", re.I)),
    ("sampling",   re.compile(r"sampl|topk|top_k|top_p|softmax_warp|multinomial|"
                              r"argmax|Sort|cub::Device|radix|scan", re.I)),
    ("norm_act",   re.compile(r"rms_norm|layer_norm|rmsnorm|layernorm|silu|swiglu|"
                              r"gelu|act_and_mul|l2norm|group_norm|sigmoid", re.I)),
    ("rope_emb",   re.compile(r"rope|rotary|embedding|Indexing|index_select|"
                              r"gather|scatter|EmbeddingBag", re.I)),
    ("input_prep", re.compile(r"input_prep|prepare_|write_req|set_kv|store_kv|"
                              r"cache_kv|copy_kv|kv_cache", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized_|CatArrayBatched|"
                               r"unrolled_|reduce_kernel|fill_|Copy|copy_|"
                               r"triu|masked|arange|clamp|where|add_|mul_", re.I)),
]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def bucket(name):
    for b, rx in RULES:
        if rx.search(name):
            return b
    return "unclassified"


def union_len(ivs):
    if not ivs:
        return 0.0
    ivs = sorted(ivs)
    total = 0.0
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    return total + (ce - cs)


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    tr = load(path)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr

    dev = []          # (ts, ts+dur, name, cat, stream)
    for e in evs:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat in DEVICE_CATS and cat != "gpu_user_annotation":
            ts = float(e["ts"])
            dur = float(e.get("dur", 0.0))
            dev.append((ts, ts + dur, e.get("name", "?"), cat,
                        e.get("args", {}).get("stream")))
    dev.sort()
    if not dev:
        print("no device events"); return

    # --- step delimiter ---
    counts = defaultdict(int)
    for _, _, n, c, _ in dev:
        if c == "kernel":
            counts[n] += 1
    delim = None
    for cand in ("fused_decode_input_prep_kernel",):
        for n in counts:
            if cand in n:
                delim = n
                break
        if delim:
            break
    if delim is None:
        # fall back: the kernel with the lowest count >= 5 that divides evenly
        cands = sorted(counts.items(), key=lambda kv: kv[1])
        print("delimiter not found; rarest kernels:", cands[:10])
        return
    marks = [t[0] for t in dev if t[2] == delim]
    marks.sort()
    nsteps = len(marks) - 1
    print(f"=== {label} ===")
    print(f"device events: {len(dev)}   delimiter '{delim}' x{len(marks)}  "
          f"-> {nsteps} intervals")
    if nsteps < 3:
        print("too few steps"); return

    # drop first and last interval (profiler warm-up / truncation)
    walls = [(marks[i + 1] - marks[i]) / 1000.0 for i in range(nsteps)]
    lo, hi = 1, nsteps - 1
    use = list(range(lo, hi))
    if len(use) < 3:
        use = list(range(nsteps))
    ws = [walls[i] for i in use]
    mean_w = sum(ws) / len(ws)
    sd_w = (sum((x - mean_w) ** 2 for x in ws) / max(1, len(ws) - 1)) ** 0.5
    med = sorted(ws)[len(ws) // 2]
    print(f"steps used: {len(use)}   wall/step: mean {mean_w:.3f} ms  "
          f"sd {sd_w:.3f}  median {med:.3f}  min {min(ws):.3f} max {max(ws):.3f}")

    # --- per-step decomposition ---
    agg = defaultdict(float)          # bucket -> ms summed over used steps
    agg_n = defaultdict(int)
    per_kernel = defaultdict(lambda: [0.0, 0])
    busy_k = 0.0                      # union of kernel-only
    busy_all = 0.0                    # union incl. memcpy/memset
    ngaps = 0
    gap_ms = 0.0
    nlaunch = 0
    streams = set()

    for i in use:
        s, e = marks[i], marks[i + 1]
        win = [d for d in dev if d[0] >= s and d[0] < e]
        kern = [(a, b) for a, b, n, c, st in win if c == "kernel"]
        alld = [(a, b) for a, b, n, c, st in win]
        busy_k += union_len(kern) / 1000.0
        busy_all += union_len(alld) / 1000.0
        for a, b, n, c, st in win:
            streams.add(st)
            if c == "kernel":
                bk = bucket(n)
                agg[bk] += (b - a) / 1000.0
                agg_n[bk] += 1
                per_kernel[n][0] += (b - a) / 1000.0
                per_kernel[n][1] += 1
                nlaunch += 1
            else:
                agg["memcpy_memset"] += (b - a) / 1000.0
                agg_n["memcpy_memset"] += 1
        # inter-kernel gaps within the step (kernel stream only)
        ks = sorted(kern)
        prev_end = None
        for a, b in ks:
            if prev_end is not None and a > prev_end:
                ngaps += 1
                gap_ms += (a - prev_end) / 1000.0
            if prev_end is None or b > prev_end:
                prev_end = b

    n = len(use)
    print(f"streams seen: {sorted(str(x) for x in streams)}")
    print(f"GPU busy (kernels only, union): {busy_k/n:.3f} ms  "
          f"({100*busy_k/n/mean_w:.1f}%)")
    print(f"GPU busy (incl copy/memset)   : {busy_all/n:.3f} ms  "
          f"({100*busy_all/n/mean_w:.1f}%)")
    print(f"GPU idle (host-attributable)  : {mean_w-busy_all/n:.3f} ms  "
          f"({100*(mean_w-busy_all/n)/mean_w:.1f}%)")
    print(f"kernel launches/step: {nlaunch/n:.1f}   "
          f"inter-kernel gaps/step: {ngaps/n:.1f}  "
          f"total {gap_ms/n:.3f} ms  mean {1000*gap_ms/max(1,ngaps):.2f} us")
    print()
    print(f"{'bucket':<16}{'ms/step':>10}{'% step':>9}{'kernels/step':>14}")
    tot = 0.0
    for b in sorted(agg, key=lambda k: -agg[k]):
        tot += agg[b]
        print(f"{b:<16}{agg[b]/n:>10.3f}{100*agg[b]/n/mean_w:>8.1f}%"
              f"{agg_n[b]/n:>14.1f}")
    print(f"{'SUM(serial)':<16}{tot/n:>10.3f}{100*tot/n/mean_w:>8.1f}%")
    print()
    print("top kernels by time/step:")
    for k in sorted(per_kernel, key=lambda k: -per_kernel[k][0])[:22]:
        ms, cnt = per_kernel[k]
        print(f"  {ms/n:>8.3f} ms {cnt/n:>7.1f}x  [{bucket(k):<12}] {k[:88]}")
    unc = [k for k in per_kernel if bucket(k) == "unclassified"]
    if unc:
        print("\nunclassified kernels:")
        for k in sorted(unc, key=lambda k: -per_kernel[k][0])[:20]:
            print(f"  {per_kernel[k][0]/n:>8.3f} ms {per_kernel[k][1]/n:>7.1f}x  {k[:100]}")


if __name__ == "__main__":
    main()
