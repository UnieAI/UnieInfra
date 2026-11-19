#!/bin/bash
cd /home/smc/workspace/roy
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> logs/gilspy.log.server 2>&1
if scripts/wait_ready.sh 3000 >> logs/gilspy.log.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 2048 > /dev/null 2>&1 &
  sleep 45
  P=$(docker exec ts-dev pgrep -f "tokenspeed::scheduler_tp0" | head -1)
  docker exec -u root ts-dev py-spy record --pid $P --threads --idle --native -r 250 -d 10 -f raw -o /logs/spy_all.raw > /dev/null 2>&1
  docker exec -u root ts-dev py-spy record --pid $P --threads --gil --native -r 250 -d 10 -f raw -o /logs/spy_gil.raw > /dev/null 2>&1
  wait
fi
scripts/stop_servers.sh
echo "GILSPY DONE" > logs/gilspy.log
