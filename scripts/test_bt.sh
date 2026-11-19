#!/bin/bash
cd /home/smc/workspace/roy
T=bt; LOG=logs/$T.log
scripts/stop_servers.sh
TS_EXP_TIME_BT=1 TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
  grep BTTIME logs/unieinfra-server.log | tail -20 >> $LOG
fi
scripts/stop_servers.sh
echo "BT DONE $(date -u +%FT%TZ)" >> $LOG
