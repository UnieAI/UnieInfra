#!/bin/bash
# Prefill FFN comm overlap A/B: 104K prefill x3 + needle, chunks 1/2/4.
cd /home/smc/workspace/roy
LOG=logs/overlap-iris-ab.log
until grep -q "H2H DONE" logs/head2head.log 2>/dev/null; do sleep 30; done
for ch in 1 2 4; do
  scripts/stop_servers.sh
  TS_EXP_FFN_OVERLAP_CHUNKS=$ch TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=64 scripts/serve_unieinfra.sh 16 >> $LOG.server 2>&1
  if ! scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then echo "ov$ch SERVER FAILED" >> $LOG; continue; fi
  python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 3 2>&1 | grep -E "round|error" | sed "s/^/ov$ch /" >> $LOG
  python3 scripts/needle_test.py 2>&1 | tail -1 | sed "s/^/ov$ch /" >> $LOG
  echo "ov$ch segfaults=$(grep -ac -E 'Segfault|Traceback' $(readlink -f logs/unieinfra-server.log))" >> $LOG
done
scripts/stop_servers.sh
echo "OVERLAP DONE $(date -u +%FT%TZ)" >> $LOG
