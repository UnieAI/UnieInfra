#!/bin/bash
# Shared-group reads through the page cache: restart on the existing dedup L3, replay needle.
set -u; cd /home/smc/workspace/roy
OUT=logs/l3e2e/needle_dd.jsonl
sync; docker exec -u root ts-dev bash -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "drop_caches not permitted"
L=$(ENGINE_WT=.wt-l3dd ENABLE_DSPARK=1 IO_THREADS=16 scripts/serve_l3.sh dd3)
until grep -qE "Scheduler hit an exception|serve ready" $L; do sleep 10; done
python3 scripts/l3_e2e.py needle_dd_l3_buffered $OUT 24 24000 > /dev/null
python3 - <<'PY'
import json, statistics as st
R = [json.loads(l) for l in open("logs/l3e2e/needle_dd.jsonl")]
for p in ["needle_dd_l3", "needle_dd_l3_buffered"]:
    d = [r for r in R if r["phase"] == p]; t = [r["ttft_s"] for r in d[1:]]
    print(p, "correct", sum(bool(r["hit"]) for r in d), "/", len(d), "ttft median %.3f" % st.median(t))
PY
grep "\[L3\] prefetch pages" $L | grep -o "elapsed_ms=[0-9.]*" | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END {print "prefetch ms per rank: median", a[int(NR/2)+1], "n", NR}'
echo DONE $L
