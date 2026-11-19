#!/bin/bash
# Re-measure the vLLM c1 cell of the "2x max-num-seqs" series (the earlier run had ITL 36 ms vs 6 ms: degraded).
cd /home/smc/workspace/roy
while pgrep -f "run_round.sh k8[23]" >/dev/null || pgrep -f "test_k8[23].sh" >/dev/null; do sleep 10; done
STRESS=1 VLLM_MAX_NUM_SEQS=2 scripts/run_round.sh stressbig 300 vllm "1"
echo "K84 DONE $(date -u +%FT%TZ)" >> logs/round-stressbig.log
