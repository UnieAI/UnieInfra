#!/bin/bash
cd /home/smc/workspace/roy
while pgrep -f "[r]un_round.sh|scripts/[t]est_k|[h]bm_server.sh" >/dev/null; do sleep 60; done
T=q8; LOG=logs/$T.log
scripts/stop_servers.sh
TOKENSPEED_IRIS_Q8_ALLREDUCE=1 TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 1500 >> $LOG.server 2>&1; then
  python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 3 2>&1 | grep round | sed "s/^/$T /" >> $LOG
  python3 scripts/needle_test.py 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
  scripts/run_gsm8k.sh unieinfra-$T >> $LOG.server 2>&1
  python3 -c "import json,glob;f=sorted(glob.glob('results/gsm8k/unieinfra-$T-*/*/reports/*/gsm8k.json'))[-1];j=json.load(open(f));print('$T gsm8k',[m.get('score') for m in j.get('metrics',[])][:1])" >> $LOG
else echo "$T SERVER FAILED" >> $LOG; fi
scripts/stop_servers.sh
echo "Q8 DONE $(date -u +%FT%TZ)" >> $LOG
