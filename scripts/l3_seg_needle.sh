#!/bin/bash
set -u; cd /home/smc/workspace/roy
OUT=logs/l3e2e/needle_seg.jsonl
python3 scripts/l3_e2e.py needle_seg_cold $OUT 24 24000 > /dev/null
L=$(ENGINE_WT=.wt-l3seg ENABLE_DSPARK=1 IO_THREADS=16 scripts/serve_l3.sh seg3)
until grep -qE "Scheduler hit an exception|serve ready" $L; do sleep 10; done
python3 scripts/l3_e2e.py needle_seg_l3 $OUT 24 24000 > /dev/null
python3 - <<'PY'
import json, statistics as st
R = [json.loads(l) for l in open("logs/l3e2e/needle_seg.jsonl")]
for p in ["needle_seg_cold", "needle_seg_l3"]:
    d = [r for r in R if r["phase"] == p]; t = [r["ttft_s"] for r in d[1:]]
    print(p, "correct", sum(bool(r["hit"]) for r in d), "/", len(d), "ttft median %.3f" % st.median(t))
PY
grep "Prefill batch" $L | grep -v "#new-token: 1," | grep -o "#cached-token: [0-9]*" | sort | uniq -c | head -2
echo DONE $L
