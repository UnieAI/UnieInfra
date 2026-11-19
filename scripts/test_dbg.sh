#!/bin/bash
cd /home/smc/workspace/roy
scripts/stop_servers.sh
TS_EXP_TIME_BT=1 TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> logs/dbg.log.server 2>&1
if scripts/wait_ready.sh 3000 >> logs/dbg.log.server 2>&1; then
  timeout 120 python3 scripts/bench_decode_shared.py --n 64 --max-tokens 256 > /dev/null 2>&1
  grep -h ZERODBG logs/unieinfra-server.log | sort | uniq -c > logs/dbg.log
fi
scripts/stop_servers.sh
echo "DBG DONE" >> logs/dbg.log
