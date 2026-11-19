#!/bin/bash
# Prefill GPU profile on the current build (155K prompt), after other benchmark rounds finish.
cd /home/smc/workspace/roy
while true; do
  if ! pgrep -f "[r]un_round.sh|scripts/[t]est_|[h]bm_server.sh" >/dev/null; then
    sleep 120; pgrep -f "[r]un_round.sh|scripts/[t]est_|[h]bm_server.sh" >/dev/null || break
  fi
  sleep 60
done
LOG=logs/profpf.log
echo "START $(date -u +%FT%TZ)" >> $LOG
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 1500 >> $LOG.server 2>&1; then
  python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 1 2>&1 | grep round | sed "s/^/warm /" >> $LOG
  curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d '{"output_dir": "/logs/prof-prefill-new", "activities": ["GPU"]}' >> $LOG.server
  python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 1 2>&1 | grep round | sed "s/^/prof /" >> $LOG
  curl -s -X POST localhost:8000/stop_profile >> $LOG.server
else echo "SERVER FAILED" >> $LOG; fi
scripts/stop_servers.sh
echo "PROFPF DONE $(date -u +%FT%TZ)" >> $LOG
