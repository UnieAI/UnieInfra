#!/bin/bash
cd /home/smc/workspace/roy
T=zx2; LOG=logs/$T.log
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 2048 > /dev/null 2>&1 &
  sleep 45
  curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d '{"output_dir": "/logs/prof-zx2", "activities": ["GPU"]}' >> $LOG.server
  sleep 4; curl -s -X POST localhost:8000/stop_profile >> $LOG.server; wait
  python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 2 2>&1 | grep round | sed "s/^/$T /" >> $LOG
  python3 scripts/needle_test.py 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
  scripts/run_gsm8k.sh unieinfra-$T >> $LOG.server 2>&1
  python3 -c "import json,glob;f=sorted(glob.glob('results/gsm8k/unieinfra-$T-*/*/reports/*/gsm8k.json'))[-1];j=json.load(open(f));print('$T gsm8k',[m.get('score') for m in j.get('metrics',[])][:1])" >> $LOG
else
  echo "$T SERVER FAILED" >> $LOG
fi
scripts/stop_servers.sh
echo "ZX2 DONE $(date -u +%FT%TZ)" >> $LOG
