#!/bin/bash
# Phases on the running server; then restart (same L3 dir) and replay.
set -u
cd /home/smc/workspace/roy
OUT=logs/l3e2e/run.jsonl
python3 scripts/l3_e2e.py cold  $OUT 24 24000 > /dev/null
python3 scripts/l3_e2e.py warm  $OUT 24 24000 > /dev/null
docker exec -u root ts-dev bash -c 'du -sh /workspace/.l3cache; find /workspace/.l3cache -name "*.kv" | wc -l'
L=$(scripts/serve_l3.sh c)
until grep -qE "Traceback|serve ready" $L; do sleep 10; done
grep -E "Traceback|rebuilt|index" $L | head -5
python3 scripts/l3_e2e.py l3restart $OUT 24 24000 > /dev/null
echo DONE $L
