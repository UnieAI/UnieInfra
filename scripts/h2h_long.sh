#!/bin/bash
# Long-context decode head-to-head on a shared 130K prefix (n=16/64), with a
# UnieInfra GPU profile of steady decode; then the multi-turn cache test.
cd /home/smc/workspace/roy
LOG=logs/head2head-long.log
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/unie /" >> $LOG
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/unie /" >> $LOG &
  sleep 60
  curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' \
    -d '{"output_dir": "/logs/prof-decode-long", "num_steps": 20, "activities": ["GPU"]}' >> $LOG; echo >> $LOG
  wait
fi
scripts/stop_servers.sh
VLLM_MAX_NUM_SEQS=128 scripts/serve_vllm.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/vllm /" >> $LOG
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/vllm /" >> $LOG
fi
scripts/stop_servers.sh
echo "H2HLONG DONE $(date -u +%FT%TZ)" >> $LOG
