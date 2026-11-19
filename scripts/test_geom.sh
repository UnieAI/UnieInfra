#!/bin/bash
cd /home/smc/workspace/roy
LOG=logs/geom.log
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/geom /" >> $LOG
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/geom /" >> $LOG
  python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 2 2>&1 | grep round | sed "s/^/geom /" >> $LOG
  python3 scripts/needle_test.py 2>&1 | tail -1 | sed "s/^/geom /" >> $LOG
  scripts/run_gsm8k.sh unieinfra-geom >> $LOG.server 2>&1
  python3 -c "import json,glob;f=sorted(glob.glob('results/gsm8k/unieinfra-geom-*/*/reports/*/gsm8k.json'))[-1];j=json.load(open(f));print('geom gsm8k',[m.get('score') for m in j.get('metrics',[])][:1])" >> $LOG
else
  echo "geom SERVER FAILED" >> $LOG
fi
scripts/stop_servers.sh
echo "GEOM DONE $(date -u +%FT%TZ)" >> $LOG
