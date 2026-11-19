#!/bin/bash
# Cross-request prefix sharing probe (diagnostic). Starts a c4-config server, sends the probe, prints cached tokens.
cd /home/smc/workspace/roy
while pgrep -f "test_k8[0-9].sh" >/dev/null || pgrep -f "run_round.sh" >/dev/null; do sleep 10; done
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 MAX_NUM_SEQS=8 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/serve_unieinfra.sh 4
scripts/wait_ready.sh 3000
python3 scripts/prefix_share_probe.py 8000 > logs/prefix_probe.out 2>&1
sleep 3
grep -E "Prefill batch" logs/unieinfra-server.log | tail -12 | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-230 >> logs/prefix_probe.out
scripts/stop_servers.sh
echo "PROBE DONE $(date -u +%FT%TZ)" >> logs/prefix_probe.out
