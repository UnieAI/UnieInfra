#!/bin/bash
cd /home/smc/workspace/roy
# Wait until no other benchmark round/test script is running, twice in a row 2 min apart.
while true; do
  if ! pgrep -f "[r]un_round.sh|scripts/[t]est_k|[h]bm_server.sh" >/dev/null; then
    sleep 120
    pgrep -f "[r]un_round.sh|scripts/[t]est_k|[h]bm_server.sh" >/dev/null || break
  fi
  sleep 60
done
echo "START $(date -u +%FT%TZ)" >> logs/round-k80b.log
scripts/test_k80.sh
