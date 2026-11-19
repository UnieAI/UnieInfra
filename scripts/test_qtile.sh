#!/bin/bash
cd /home/smc/workspace/roy
LOG=logs/qtile.log
for t in 384 256; do
  scripts/stop_servers.sh
  TS_EXP_DECODE_INDEX_QUERY_TILE=$t TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  scripts/wait_ready.sh 3000 >> $LOG.server 2>&1 || { echo "t$t FAILED" >> $LOG; continue; }
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/t$t /" >> $LOG
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/t$t /" >> $LOG
  python3 scripts/needle_test.py 2>&1 | tail -1 | sed "s/^/t$t /" >> $LOG
done
scripts/stop_servers.sh
echo "QTILE DONE $(date -u +%FT%TZ)" >> $LOG
