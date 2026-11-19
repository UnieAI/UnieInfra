#!/bin/bash
# Same micro-tests on both engines: one 104K prefill (x3), decode speed at n=1/16/64.
cd /home/smc/workspace/roy
LOG=logs/head2head.log
until grep -q "LPM DONE" logs/round-lpm.log 2>/dev/null; do sleep 30; done
run_tests() {
  python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 3 2>&1 | sed "s/^/$1 /" >> $LOG
  for n in 1 16 64; do
    python3 scripts/bench_decode.py --n $n --tokens 2000 --max-tokens 512 2>&1 | tail -2 | sed "s/^/$1 n=$n /" >> $LOG
  done
}
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG 2>&1
scripts/wait_ready.sh 3000 >> $LOG 2>&1 && run_tests unie
scripts/stop_servers.sh
VLLM_MAX_NUM_SEQS=128 scripts/serve_vllm.sh 64 >> $LOG 2>&1
scripts/wait_ready.sh 3000 >> $LOG 2>&1 && run_tests vllm
scripts/stop_servers.sh
echo "H2H DONE $(date -u +%FT%TZ)" >> $LOG
