#!/bin/bash
# UnieInfra long-context decode profile (explicit stop) + vLLM shared-prefix decode bench.
cd /home/smc/workspace/roy
LOG=logs/head2head-long.log
until grep -q "CACHECMP DONE" logs/multiturn-cache.log 2>/dev/null; do sleep 20; done
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 2048 2>&1 | tail -1 | sed "s/^/unie-prof /" >> $LOG &
  sleep 45
  curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' \
    -d '{"output_dir": "/logs/prof-decode-long", "activities": ["GPU"]}' >> $LOG; echo >> $LOG
  sleep 4
  curl -s -X POST localhost:8000/stop_profile >> $LOG; echo >> $LOG
  wait
  sleep 20
fi
scripts/stop_servers.sh
VLLM_MAX_NUM_SEQS=128 scripts/serve_vllm.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1 | sed "s/^/vllm /" >> $LOG
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/vllm /" >> $LOG
fi
scripts/stop_servers.sh
echo "PROF DONE $(date -u +%FT%TZ)" >> $LOG
