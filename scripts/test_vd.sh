#!/bin/bash
# DSpark verify-drafts A/B on the 130K shared-prefix decode bench.
cd /home/smc/workspace/roy
LOG=logs/vd.log
for V in 3 5; do
  T=vd$V
  scripts/stop_servers.sh
  TOKENSPEED_DSPARK_VERIFY_DRAFTS=$V TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  if scripts/wait_ready.sh 1500 >> $LOG.server 2>&1; then
    for i in 1 2; do
      python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
      python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
    done
    sed 's/\x1b\[[0-9;]*m//g' $(readlink -f logs/unieinfra-server.log) | grep "Decode batch" | tail -6 | grep -oE "#running-req: [0-9]+|avg_accept_len: [0-9.]+" | paste - - | sed "s/^/$T /" >> $LOG
    [ $V = 3 ] && python3 scripts/needle_test.py 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
  else echo "$T SERVER FAILED" >> $LOG; fi
done
scripts/stop_servers.sh
echo "VD DONE $(date -u +%FT%TZ)" >> $LOG
