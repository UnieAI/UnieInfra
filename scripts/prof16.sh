#!/bin/bash
# 16-user 130K decode profiles: UnieInfra (0f528991) vs vLLM.
cd /home/smc/workspace/roy
LOG=logs/prof16.log
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 1500 >> $LOG.server 2>&1; then

  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 8192 2>&1 | tail -1 | sed "s/^/unie-prof /" >> $LOG &
  sleep 25
  curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d '{"output_dir": "/logs/prof16b-unie", "activities": ["GPU"]}' >> $LOG.server
  sleep 4; curl -s -X POST localhost:8000/stop_profile >> $LOG.server; wait
  sed 's/\x1b\[[0-9;]*m//g' $(readlink -f logs/unieinfra-server.log) | grep "Decode batch" | tail -4 | grep -oE "#running-req: [0-9]+|avg_accept_len: [0-9.]+" | paste - - >> $LOG
fi
scripts/stop_servers.sh
VLLM_USE_RUST_FRONTEND=0 VLLM_PROFILER_CONFIG='{"profiler":"torch","torch_profiler_dir":"/logs/prof16b-vllm","torch_profiler_activities":["CUDA"],"torch_profiler_with_stack":false,"active_iterations":40}' VLLM_MAX_NUM_SEQS=128 scripts/serve_vllm.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then

  python3 scripts/bench_decode_shared.py --n 16 --max-tokens 8192 2>&1 | tail -1 | sed "s/^/vllm-prof /" >> $LOG &
  sleep 25
  curl -s -X POST localhost:8000/start_profile >> $LOG.server; sleep 4; curl -s -X POST localhost:8000/stop_profile >> $LOG.server; wait; sleep 30
  docker logs bench-server 2>&1 | grep -iE "SpecDecoding metrics" | tail -3 >> $LOG
fi
scripts/stop_servers.sh
echo "PROF16B DONE $(date -u +%FT%TZ)" >> $LOG
