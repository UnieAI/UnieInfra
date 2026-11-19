#!/bin/bash
cd /home/smc/workspace/roy
LOG=logs/iris32.log
for p in 32; do
  scripts/stop_servers.sh
  TOKENSPEED_IRIS_PULL_MAX_PROGRAMS=$p TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  scripts/wait_ready.sh 3000 >> $LOG.server 2>&1 || { echo "p$p FAILED" >> $LOG; continue; }
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/p$p /" >> $LOG
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/p$p /" >> $LOG
  python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 2 2>&1 | grep round | sed "s/^/p$p /" >> $LOG
  python3 scripts/needle_test.py 2>&1 | tail -1 | sed "s/^/p$p /" >> $LOG
done
export TOKENSPEED_IRIS_PULL_MAX_PROGRAMS=32 TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh iris32 300 unieinfra 64
STRESS=1 MAX_NUM_SEQS=64 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh iris32 300 unieinfra 16
echo "IRIS32 DONE $(date -u +%FT%TZ)" >> $LOG
