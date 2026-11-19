#!/bin/bash
# Server-side HBM counters per kernel (eager mode), 16- and 64-user 130K decode.
cd /home/smc/workspace/roy
LOG=logs/hbm_server.log
scripts/stop_servers.sh
docker exec -u root ts-dev rm -rf /logs/hbmprof
EXTRA_SERVE_ARGS="--prefill-coalesce-ms 0" TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra_pmc.sh 64 >> $LOG 2>&1
if scripts/wait_ready.sh 3000 >> $LOG 2>&1; then
  date -u +"BENCH16 %s.%N" >> $LOG
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 256 2>&1 | tail -1 | sed "s/^/pmc /" >> $LOG
  date -u +"BENCH64 %s.%N" >> $LOG
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 256 2>&1 | tail -1 | sed "s/^/pmc /" >> $LOG
  date -u +"BENCHEND %s.%N" >> $LOG
else echo "SERVER FAILED" >> $LOG; fi
scripts/stop_servers.sh
sleep 20
touch logs/hbm_server.done
echo "HBM DONE $(date -u +%FT%TZ)" >> $LOG
