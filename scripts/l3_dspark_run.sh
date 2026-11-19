#!/bin/bash
# DSpark + Host L2 + file L3: needle cold -> warm (L2) -> restart -> L3, with acceptance per phase.
set -u
cd /home/smc/workspace/roy
OUT=logs/l3e2e/needle_dspark.jsonl
acc() { grep -o "Accept_num_tokens_avg: [0-9.]*" "$1" | tail -24 | awk '{s+=$2;n++} END {printf "accept_avg %.2f over %d\n", s/n, n}'; }
L=$(ls -t logs/unieinfra-l3-dspark2-* | head -1)
python3 scripts/l3_e2e.py needle_ds_cold $OUT 24 24000 > /dev/null; echo -n "cold "; acc $L
python3 scripts/l3_e2e.py needle_ds_warm $OUT 24 24000 > /dev/null; echo -n "warm "; acc $L
L=$(ENABLE_DSPARK=1 IO_THREADS=16 scripts/serve_l3.sh dspark3)
until grep -qE "Scheduler hit an exception|serve ready" $L; do sleep 10; done
grep -E "Scheduler hit an exception" $L | head -1
python3 scripts/l3_e2e.py needle_ds_l3 $OUT 24 24000 > /dev/null; echo -n "l3 "; acc $L
python3 - <<'PY'
import json, statistics as st
R = [json.loads(l) for l in open("logs/l3e2e/needle_dspark.jsonl")]
for p in ["needle_ds_cold", "needle_ds_warm", "needle_ds_l3"]:
    d = [r for r in R if r["phase"] == p]
    t = [r["ttft_s"] for r in d[1:]]
    print(p, "correct", sum(r["hit"] for r in d), "/", len(d), "ttft median %.3f (first %.3f)" % (st.median(t), d[0]["ttft_s"]))
PY
grep "Prefill batch" $L | grep -v "#new-token: 1," | grep -o "#cached-token: [0-9]*" | sort | uniq -c | head -3
echo DONE $L
