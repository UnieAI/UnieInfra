#!/bin/bash
# Stress first (user priority), then the remaining official UnieInfra cells.
# 1) let the running official c128 point finish, then stop that round before c64
# 2) stress c64/c128/c32/c16/c1, UnieInfra and vLLM, official deployment parameters
# 3) stress vLLM c64/c128/c32/c16/c1 with max-num-seqs raised to 2x CONC (as UnieInfra)
# 4) saturation points (c256; vLLM max-num-seqs sweep at c128)
# 5) official UnieInfra c64, c1, c4, c16, c32
ROUND_PID=${1:?pid of the running c128/c64 round}
cd /home/smc/workspace/roy
until grep -qE "unieinfra c128: (done|SERVER FAILED)" logs/round-comm.log; do sleep 30; done
kill "$ROUND_PID"
sleep 5
STRESS=1 scripts/run_round.sh stress 300 "unieinfra vllm" "64 128 32 16 1"
for c in 64 128 32 16 1; do
  STRESS=1 VLLM_MAX_NUM_SEQS=$(( 2 * c )) scripts/run_round.sh stressbig 300 vllm "$c"
done
# Saturation: where does throughput stop rising?
#  - concurrency past c128: c256 for UnieInfra and for vLLM with 2x CONC seqs
#  - vLLM's own knob at fixed c128: max-num-seqs 64 and 128 (32 and 256 run above)
STRESS=1 scripts/run_round.sh stress 300 unieinfra "256"
STRESS=1 VLLM_MAX_NUM_SEQS=512 scripts/run_round.sh stressbig 300 vllm "256"
for s in 64 128; do
  STRESS=1 VLLM_MAX_NUM_SEQS=$s scripts/run_round.sh "stressseq$s" 300 vllm "128"
done
scripts/run_round.sh comm 900 unieinfra "64 1 4 16 32"
echo "ALL DONE $(date -u +%FT%TZ)" >> logs/round-comm.log
