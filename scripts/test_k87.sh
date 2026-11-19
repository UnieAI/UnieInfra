#!/bin/bash
# Re-run of the vLLM "2x max-num-seqs" c1 cell. The first try (k84) found port 8000 held by a leftover UnieInfra server.
cd /home/smc/workspace/roy
while pgrep -f "test_k8[56].sh" >/dev/null || pgrep -f "run_round.sh" >/dev/null; do sleep 10; done
scripts/stop_servers.sh
sleep 20
STRESS=1 VLLM_MAX_NUM_SEQS=2 scripts/run_round.sh stressbig 300 vllm "1"
echo "K87 DONE $(date -u +%FT%TZ)" >> logs/round-stressbig.log
