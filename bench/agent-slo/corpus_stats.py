"""Measure reuse structure of a bench-data jsonl corpus.

same-session  : LCP(turn_k, turn_k+1) / len(turn_k)      -- containment of the earlier turn
cross-session : LCP(a, b) / min(len(a), len(b))          -- shared preamble
Both computed in TOKEN space (prefix cache works on token blocks) and char space.
"""

import hashlib
import json
import random
import statistics as st
import sys

from transformers import AutoTokenizer


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main(
    path,
    nsess,
    tokname="/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    sess_key=None,
):
    rows = [json.loads(l) for l in open(path)]
    prompts = [r["prompt"] for r in rows]
    n = len(prompts)
    uniq = len({hashlib.sha256(p.encode()).hexdigest() for p in prompts})
    print(f"file            : {path}")
    print(f"requests        : {n}")
    print(
        f"unique prompts  : {uniq}   ({'OK all unique' if uniq==n else '!! DUPLICATES: %d'%(n-uniq)})"
    )
    tok = AutoTokenizer.from_pretrained(tokname, use_fast=True, local_files_only=True)
    ids = tok(prompts, add_special_tokens=False)["input_ids"]
    tl = sorted(len(x) for x in ids)
    cl = sorted(len(p) for p in prompts)

    def dist(v, unit):
        q = lambda p: v[int(p * (len(v) - 1))]
        print(
            f"  {unit:6s} min={v[0]:>8,} p25={q(.25):>8,} median={q(.5):>8,} p75={q(.75):>8,} p95={q(.95):>8,} max={v[-1]:>8,} mean={sum(v)//len(v):>8,}"
        )

    print("length distribution:")
    dist(cl, "chars")
    dist(tl, "tokens")

    # session assignment: explicit key, or round-robin modulo
    if sess_key and sess_key in rows[0]:
        sess = {}
        for i, r in enumerate(rows):
            sess.setdefault(r[sess_key], []).append(i)
        groups = list(sess.values())
    else:
        groups = [[i for i in range(n) if i % nsess == s] for s in range(nsess)]
    print(
        f"sessions        : {len(groups)}  (sizes {sorted({len(g) for g in groups})})"
    )

    same_t, same_c = [], []
    for g in groups:
        for a, b in zip(g, g[1:]):
            if len(ids[a]):
                same_t.append(lcp(ids[a], ids[b]) / len(ids[a]))
            if len(prompts[a]):
                same_c.append(lcp(prompts[a], prompts[b]) / len(prompts[a]))
    cross_t, cross_c = [], []
    rnd = random.Random(0)
    gi = {i: k for k, g in enumerate(groups) for i in g}
    pairs = 0
    while pairs < 4000:
        a, b = rnd.randrange(n), rnd.randrange(n)
        if a == b or gi.get(a) == gi.get(b):
            continue
        m = min(len(ids[a]), len(ids[b]))
        if m:
            cross_t.append(lcp(ids[a], ids[b]) / m)
        m2 = min(len(prompts[a]), len(prompts[b]))
        if m2:
            cross_c.append(lcp(prompts[a], prompts[b]) / m2)
        pairs += 1

    def rep(name, v):
        v = sorted(v)
        q = lambda p: v[int(p * (len(v) - 1))]
        print(
            f"  {name:30s} n={len(v):>5}  median={q(.5)*100:6.1f}%  mean={sum(v)/len(v)*100:6.1f}%  p25={q(.25)*100:6.1f}%  p75={q(.75)*100:6.1f}%  zero-share={sum(1 for x in v if x<0.01)/len(v)*100:5.1f}%"
        )

    print("prefix sharing:")
    rep("same-session consecutive (tok)", same_t)
    rep("same-session consecutive (chr)", same_c)
    rep("cross-session random (tok)", cross_t)
    rep("cross-session random (chr)", cross_c)


if __name__ == "__main__":
    main(
        sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else 1,
        (
            sys.argv[3]
            if len(sys.argv) > 3
            else "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
        ),
        sys.argv[4] if len(sys.argv) > 4 else None,
    )
