#!/bin/bash
cd /home/smc/workspace/roy
LOG=logs/abi.log
for P in 84 32; do
  T=iris$P
  scripts/stop_servers.sh
  TOKENSPEED_IRIS_PULL_SMALL_PROGRAMS=$P TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  if scripts/wait_ready.sh 1500 >> $LOG.server 2>&1; then
    for i in 1 2; do
      python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
      python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
    done
    python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 2 2>&1 | grep round | sed "s/^/$T /" >> $LOG
  else echo "$T SERVER FAILED" >> $LOG; fi
done
scripts/stop_servers.sh
echo "ABI DONE $(date -u +%FT%TZ)" >> $LOG
