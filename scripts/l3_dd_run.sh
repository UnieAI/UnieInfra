#!/bin/bash
# TP-dedupe L3: needle cold -> restart -> L3 (DSpark on), disk use per rank dir.
set -u; cd /home/smc/workspace/roy
OUT=logs/l3e2e/needle_dd.jsonl
L=$(ENGINE_WT=.wt-l3dd ENABLE_DSPARK=1 IO_THREADS=16 scripts/serve_l3.sh dd1)
until grep -qE "Scheduler hit an exception|serve ready" $L; do sleep 10; done
grep -E "Scheduler hit an exception" $L | head -1
python3 scripts/l3_e2e.py needle_dd_cold $OUT 24 24000 > /dev/null
docker exec -u root ts-dev bash -c 'du -sh /workspace/.l3cache/rank* 2>/dev/null; ls /workspace/.l3cache'
L=$(ENGINE_WT=.wt-l3dd ENABLE_DSPARK=1 IO_THREADS=16 scripts/serve_l3.sh dd2)
until grep -qE "Scheduler hit an exception|serve ready" $L; do sleep 10; done
grep -E "\[L3 file\] indexed" $L | head -4 | cut -c1-200
python3 scripts/l3_e2e.py needle_dd_l3 $OUT 24 24000 > /dev/null
python3 - <<'PY'
import json, statistics as st
R = [json.loads(l) for l in open("logs/l3e2e/needle_dd.jsonl")]
for p in ["needle_dd_cold", "needle_dd_l3"]:
    d = [r for r in R if r["phase"] == p]; t = [r["ttft_s"] for r in d[1:]]
    print(p, "correct", sum(bool(r["hit"]) for r in d), "/", len(d), "ttft median %.3f" % st.median(t))
PY
grep "\[L3\] prefetch pages" $L | grep -o "elapsed_ms=[0-9.]*" | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END {print "prefetch ms per rank: median", a[int(NR/2)+1], "n", NR}'
grep "\[L3\] prefetch pages" $L | grep -o "ok=[0-9]* bytes" | sort | uniq -c | head -3
grep "Prefill batch" $L | grep -v "#new-token: 1," | grep -o "#cached-token: [0-9]*" | sort | uniq -c | head -2
echo DONE $L
