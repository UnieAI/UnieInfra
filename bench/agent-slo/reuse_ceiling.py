#!/usr/bin/env python3
"""How much prefix reuse a corpus actually contains.

A hit rate has no meaning without a denominator. 81% is excellent on a corpus
whose turns share 85% of their tokens and dishonest on one that shares 99%.
This computes the ceiling: for every request, the longest common prefix it
shares with any EARLIER request in the same session, floored to a page boundary
(a partial page cannot be reused), summed and divided by total prompt tokens.

    python3 reuse_ceiling.py corpus.jsonl [page_size]

Reports the ceiling plus the reuse distance -- how many other requests are
issued between a session's consecutive turns. That distance is what the cache
has to survive: if it exceeds what the KV pool holds, the hit rate collapses
even though the reuse is right there in the corpus.
"""

import collections
import json
import sys


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main(path, page=64):
    rows = [json.loads(l) for l in open(path)]
    total = cached = 0
    seen = collections.defaultdict(list)
    for r in rows:
        p, n = r["prompt"], r["n_prompt_tokens"]
        total += n
        best = max((lcp(p, q) for q in seen[r["session"]]), default=0)
        # 字元前綴按比例換算成 token，再向下對齊到 page
        cached += (int(n * best / len(p)) // page) * page if p else 0
        seen[r["session"]].append(p)

    # reuse distance：同一 session 連續兩輪之間隔了幾個別的請求
    pos = collections.defaultdict(list)
    for i, r in enumerate(rows):
        pos[r["session"]].append(i)
    gaps = [b - a for v in pos.values() for a, b in zip(v, v[1:])]
    gaps.sort()

    print(f"corpus            {path}")
    print(f"requests          {len(rows)} in {len(seen)} sessions")
    print(f"prompt tokens     {total:,}")
    print(
        f"reuse ceiling     {100*cached/total:.1f}%  ({cached:,} tokens are a page-aligned"
    )
    print(f"                  prefix of an earlier turn in the same session)")
    if gaps:
        print(f"reuse distance    median {gaps[len(gaps)//2]} requests, max {gaps[-1]}")
        print(
            f"                  -- the cache must hold a session's blocks across this many"
        )
        print(f"                     other requests, or the reuse above is unreachable")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64)
