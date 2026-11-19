#!/bin/bash
cd /home/smc/workspace/roy
T=sync2; LOG=logs/$T.log
scripts/stop_servers.sh
TS_EXP_ZERO_ON_EXEC=1 TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 2048 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG &
  sleep 45
  curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d '{"output_dir": "/logs/prof-sync2", "activities": ["CPU","GPU"]}' >> $LOG.server
  sleep 3; curl -s -X POST localhost:8000/stop_profile >> $LOG.server; wait
  cp logs/unieinfra-server.log logs/sync2.serverlog
fi
scripts/stop_servers.sh
echo "SYNC2 DONE $(date -u +%FT%TZ)" >> $LOG
