#!/bin/bash
# Fair sweep, official mode (recorded inter-turn gaps), 900 s per cell, one run per cell, one GPU lock for the whole sweep.
# Feature rule (user): a feature vLLM has but does not enable is not enabled in UnieInfra; a feature vLLM does not
# have is enabled in UnieInfra.
#   vLLM:      official recipe untouched (probabilistic draft + block verification, recipe token budget and running cap).
#   UnieInfra: commit 5e0db812 (main checkout only), same running cap, token budget = recipe's (c1: 16384 because 32768
#              is untested), mixed prefill/decode batches (vLLM's chunked prefill), cache-aware admission (vLLM has no
#              prefix-aware policy), coupled draft sampling (vLLM enables its own probabilistic/block variant), sampling
#              backend triton_full (honours the request temperature). OFF: prefill coalescing and decode-every-prefill
#              (vLLM has prefill_schedule_interval, default off), INT8 all-reduce (vLLM has quick-reduce, recipe sets NONE).
#   Both: GPU memory utilization 0.9, prefix caching on, same harness window. Engine order alternates per concurrency.
cd /home/smc/workspace/roy
LOCK=logs/GPU_LOCK
M=logs/aligned_sweep2.log
: > $M
until ( set -o noclobber; echo "owner=roy-d8(fair sweep) since=$(date -u +%FT%TZ) release=when 'SWEEP DONE' is written to logs/aligned_sweep2.log" > $LOCK ) 2>/dev/null; do sleep 3; done
trap 'rm -f $LOCK' EXIT
until ! pgrep -f "run_round.sh|serve_vllm.sh|serve_unieinfra.sh" >/dev/null; do sleep 2; done
echo "$(date -u +%T) lock taken; unieinfra HEAD $(git -C unieinfra rev-parse --short HEAD)" >> $M

declare -A CAP=([1]=1 [4]=8 [16]=4 [32]=8 [64]=16 [128]=32)            # vLLM recipe running caps
declare -A BUD=([1]=16384 [4]=16384 [16]=8192 [32]=8192 [64]=8192 [128]=8192)   # recipe token budgets

vcell() {  # conc tag
  echo "$(date -u +%T) start vllm c$1 (recipe, cap ${CAP[$1]})" >> $M
  scripts/run_round.sh ${2}v 900 vllm "$1"
  cp logs/vllm-last-server.log logs/vllm-${2}-c$1-server.log 2>/dev/null
  echo "$(date -u +%T) done vllm c$1; $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' logs/vllm-last-server.log | head -1)" >> $M
}
ucell() {  # conc tag
  echo "$(date -u +%T) start unieinfra c$1 (cap ${CAP[$1]}, budget ${BUD[$1]})" >> $M
  TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=${BUD[$1]} \
  MAX_NUM_SEQS=${CAP[$1]} MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-mixed-batch" \
  scripts/run_round.sh ${2}u 900 unieinfra "$1"
  echo "$(date -u +%T) done unieinfra c$1" >> $M
}
order=u
for conc in 1 4 16 32 64 128; do
  tag=FS$conc
  if [ $order = u ]; then ucell $conc $tag; vcell $conc $tag; order=v; else vcell $conc $tag; ucell $conc $tag; order=u; fi
done
echo "SWEEP DONE $(date -u +%FT%TZ)" >> $M
