#!/bin/bash
# Attention-DP bring-up: 4 GPUs as 4 attention replicas (attn TP1 x DP4), MoE TP4 with all-gather/reduce-scatter.
cd /home/smc/workspace/roy
LOG=logs/dp_bringup.log
echo "== $(date -u +%T) start" > $LOG
scripts/stop_servers.sh >> $LOG 2>&1
ATTN_DP_SIZE=4 TOKENSPEED_V41_KEEP_ALL_ROWS=1 TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 MAX_TOTAL_TOKENS=16777216 scripts/serve_unieinfra.sh 16 >> $LOG 2>&1
# Stop waiting as soon as the server reports a startup failure.
scripts/wait_ready.sh 2700 >> $LOG 2>&1 &
WAIT=$!
( while kill -0 $WAIT 2>/dev/null; do grep -q "startup failed" logs/unieinfra-server.log 2>/dev/null && kill $WAIT 2>/dev/null; sleep 5; done ) &
WATCH=$!
if wait $WAIT; then
  echo "READY $(date -u +%T)" >> $LOG
  python3 scripts/needle_test.py 2>&1 | tail -3 >> $LOG
else
  echo "NOT READY $(date -u +%T)" >> $LOG
fi
kill $WATCH 2>/dev/null
echo "BRINGUP DONE $(date -u +%T)" >> $LOG
