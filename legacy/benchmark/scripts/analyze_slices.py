#!/usr/bin/env python3
"""Extract running-batch / queue-depth / page-ratio evidence from a server-log slice.

Parses tokenspeed decode/prefill batch log lines:
  "Decode batch. #running-req: N, #pages(active/cached/total): a/c/t, page ratio: R, ... #queue-req: Q"
  "Prefill/Mix batch. ... #running-req: N, #queue-req: Q"
Reports the running-req distribution (max/median), max queue depth, max page ratio.
Usage: analyze_slices.py <slice_file> [graph_cap]
"""

import re
import statistics as st
import sys

path = sys.argv[1]
graph_cap = int(sys.argv[2]) if len(sys.argv) > 2 else None
txt = open(path, errors="ignore").read()

dec_run, dec_q, dec_pr = [], [], []
pre_run, pre_q = [], []
for line in txt.splitlines():
    if "Decode batch" in line:
        m = re.search(r"#running-req:\s*(\d+)", line)
        q = re.search(r"#queue-req:\s*(\d+)", line)
        pr = re.search(r"page ratio:\s*([\d.]+)", line)
        if m:
            dec_run.append(int(m.group(1)))
        if q:
            dec_q.append(int(q.group(1)))
        if pr:
            dec_pr.append(float(pr.group(1)))
    elif re.search(r"(Prefill|Mix) batch", line):
        m = re.search(r"#running-req:\s*(\d+)", line)
        q = re.search(r"#queue-req:\s*(\d+)", line)
        if m:
            pre_run.append(int(m.group(1)))
        if q:
            pre_q.append(int(q.group(1)))


def sm(xs):
    if not xs:
        return "n/a"
    return f"n={len(xs)} max={max(xs)} p50={int(st.median(xs))} mean={sum(xs)/len(xs):.1f} min={min(xs)}"


print(f"== {path} ==")
print(f"DECODE  running-req: {sm(dec_run)}")
print(f"DECODE  queue-req  : {sm(dec_q)}")
if dec_pr:
    print(
        f"DECODE  page-ratio : max={max(dec_pr):.2f} p50={st.median(dec_pr):.2f} mean={sum(dec_pr)/len(dec_pr):.2f}"
    )
print(f"PREFILL running-req: {sm(pre_run)}")
print(f"PREFILL queue-req  : {sm(pre_q)}")
if graph_cap and dec_run:
    eager = sum(1 for x in dec_run if x > graph_cap)
    print(
        f"EAGER decode steps (running-req > graph_cap {graph_cap}): {eager}/{len(dec_run)} ({100*eager/len(dec_run):.1f}%)"
    )
    atcap = sum(1 for x in dec_run if x >= graph_cap)
    print(
        f"AT-CAP decode steps (running-req >= {graph_cap}): {atcap}/{len(dec_run)} ({100*atcap/len(dec_run):.1f}%)"
    )
