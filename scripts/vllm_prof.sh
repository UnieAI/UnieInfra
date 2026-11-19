#!/bin/bash
# vLLM decode profile on the same shared-prefix 64-user 130K workload.
cd /home/smc/workspace/roy
LOG=logs/vllm-prof.log

scripts/stop_servers.sh
VLLM_USE_RUST_FRONTEND=0 VLLM_PROFILER_CONFIG='{"profiler":"torch","torch_profiler_dir":"/logs/prof-vllm-decode","torch_profiler_activities":["CUDA"],"torch_profiler_with_stack":false,"active_iterations":30}' VLLM_MAX_NUM_SEQS=128 scripts/serve_vllm.sh 64 >> $LOG 2>&1
if scripts/wait_ready.sh 3000 >> $LOG 2>&1; then
  python3 scripts/bench_decode_shared.py --n 64 --max-tokens 2048 2>&1 | tail -1 >> $LOG &
  sleep 45
  curl -s -w " start=%{http_code}" -X POST localhost:8000/start_profile >> $LOG; sleep 4; curl -s -w " stop=%{http_code}" -X POST localhost:8000/stop_profile >> $LOG; echo >> $LOG
  wait; sleep 30
fi
docker logs bench-server 2>&1 | grep -iE "profil" | tail -5 >> $LOG
scripts/stop_servers.sh
echo "VPROF DONE $(date -u +%FT%TZ)" >> $LOG
