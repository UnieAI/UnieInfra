#!/bin/bash
set -u; cd /home/smc/workspace/roy
OUT=logs/l3e2e/accept_dspark.jsonl
acc() { grep -o "Accept_num_tokens_avg: [0-9.]*" "$1" | tail -8 | awk '{s+=$2;n++} END {printf "accept_avg %.3f over %d\n", s/n, n}'; }
L=$(ls -t logs/unieinfra-l3-dspark3-* | head -1)
python3 scripts/l3_accept.py cold $OUT; echo -n "cold "; acc $L
L=$(ENABLE_DSPARK=1 IO_THREADS=16 scripts/serve_l3.sh dspark4)
until grep -qE "Scheduler hit an exception|serve ready" $L; do sleep 10; done
python3 scripts/l3_accept.py l3 $OUT; echo -n "l3 "; acc $L
grep "Prefill batch" $L | grep -v "#new-token: 1," | grep -o "#cached-token: [0-9]*" | sort | uniq -c | head -4
python3 -c "
import json; R=[json.loads(l) for l in open('$OUT')]
c={r['i']:r for r in R if r['phase']=='cold'}; l={r['i']:r for r in R if r['phase']=='l3'}
print('prompt_tokens', [c[i]['prompt_tokens'] for i in sorted(c)]); print('text identical', sum(c[i]['text']==l[i]['text'] for i in c), '/ 8')"
echo DONE $L
