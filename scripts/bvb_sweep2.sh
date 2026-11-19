#!/bin/bash
# Best-vs-best sweep, official mode (recorded inter-turn gaps), 900 s per cell, one run per cell, one GPU lock.
# Each engine runs its own best parameters; both are swept over the same two running caps per concurrency:
#   r = vLLM recipe cap (CONC/4), x = 2 x CONC. GPU memory utilization 0.9 for both. Engine order alternates.
#   vLLM:      official recipe untouched (probabilistic draft + block verification, recipe token budget).
#   UnieInfra: commit 5e0db812 main checkout only; coupled draft + triton_full; cache-aware admission; mixed batch OFF
#              (A/B at c4: no gain); prefill chunk 8192; no INT8 all-reduce; KV token cap 33.5M (arena follows util 0.9).
cd /home/smc/workspace/roy
LOCK=logs/GPU_LOCK
M=logs/bvb_sweep2.log
: > $M
until ( set -o noclobber; echo "owner=roy-d8(best-vs-best sweep) since=$(date -u +%FT%TZ) release=when 'BVB2 DONE' is written to logs/bvb_sweep.log" > $LOCK ) 2>/dev/null; do sleep 3; done
trap 'rm -f $LOCK' EXIT
until ! pgrep -f "run_round.sh|serve_vllm.sh|serve_unieinfra.sh" >/dev/null; do sleep 2; done
echo "$(date -u +%T) lock taken; unieinfra HEAD $(git -C unieinfra rev-parse --short HEAD)" >> $M

declare -A RCAP=([1]=1 [4]=8 [16]=4 [32]=8 [64]=16 [128]=32)

vcell() {  # conc kind(r|x) cap
  local conc=$1 kind=$2 cap=$3 tag=BV$1$2
  echo "$(date -u +%T) start vllm c$conc $kind cap=$cap" >> $M
  if [ $kind = x ]; then VLLM_MAX_NUM_SEQS=$cap scripts/run_round.sh ${tag}v 900 vllm "$conc"
  else scripts/run_round.sh ${tag}v 900 vllm "$conc"; fi
  echo "$(date -u +%T) done vllm c$conc $kind; $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' logs/vllm-last-server.log | head -1)" >> $M
}
ucell() {  # conc kind cap
  local conc=$1 kind=$2 cap=$3 tag=BV$1$2
  echo "$(date -u +%T) start unieinfra c$conc $kind cap=$cap" >> $M
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 \
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh ${tag}u 900 unieinfra "$conc"
  echo "$(date -u +%T) done unieinfra c$conc $kind" >> $M
}
order=u
for conc in 128 64 32 4 1; do
  r=${RCAP[$conc]}; x=$((2*conc))
  kinds="r x"; [ $r = $x ] && kinds="r"
  for kind in $kinds; do
    cap=$r; [ $kind = x ] && cap=$x
    if [ $order = u ]; then ucell $conc $kind $cap; vcell $conc $kind $cap; order=v; else vcell $conc $kind $cap; ucell $conc $kind $cap; order=u; fi
  done
done
echo "BVB2 DONE $(date -u +%FT%TZ)" >> $M
