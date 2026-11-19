#!/bin/bash
cd /home/smc/workspace/roy
T=spy; LOG=logs/$T.log
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1536 > /dev/null 2>&1 &
  sleep 50
  for pid in $(docker exec ts-dev pgrep -f "tokenspeed" ); do
    docker exec ts-dev py-spy dump --native --pid $pid 2>&1 | grep -q "forward_thread" && echo "PID $pid" >> $LOG && break
  done
  P=$(grep PID $LOG | tail -1 | awk '{print $2}')
  for i in $(seq 1 25); do docker exec ts-dev py-spy dump --native --pid $P 2>&1 | awk '/Thread .*forward|_run \(tokenspeed\/runtime\/execution\/forward_thread/{f=1} f{print} /^$/{if(f)exit}' | head -30 >> $LOG; echo ---- >> $LOG; done
  wait
fi
scripts/stop_servers.sh
echo "SPY DONE $(date -u +%FT%TZ)" >> $LOG
