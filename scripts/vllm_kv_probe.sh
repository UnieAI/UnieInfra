#!/bin/bash
# Start vLLM with the algorithm-aligned launch settings, keep its log, report the KV cache size it chose.
cd /home/smc/workspace/roy
L=logs/vllm_kv_probe.log
echo "== $(date -u +%T) start" > $L
while pgrep -f "run_round.sh" >/dev/null; do sleep 2; done
scripts/stop_servers.sh >> $L 2>&1
VLLM_ALIGN=1 VLLM_MAX_NUM_SEQS=4 scripts/serve_vllm.sh 16 >> $L 2>&1
if scripts/wait_ready.sh 3000 >> $L 2>&1; then
  docker logs bench-server > logs/vllm-L0-server.log 2>&1
  grep -oE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|speculative_config=[^ ]*|max_num_batched_tokens=[0-9]+|max_num_seqs=[0-9]+" logs/vllm-L0-server.log | sort -u >> $L
  echo "READY" >> $L
else
  echo "NOT READY" >> $L
fi
scripts/stop_servers.sh >> $L 2>&1
echo "KVPROBE DONE $(date -u +%T)" >> $L
