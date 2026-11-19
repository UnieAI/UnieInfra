#!/bin/bash
# Level 0 (algorithm-aligned) sweep, official mode (recorded inter-turn gaps), 900 s per cell, one run per cell.
# Per concurrency, vLLM first (its launch settings fix the KV cache size), then UnieInfra with the SAME running
# cap and the SAME KV token budget. c16 is already covered by the pilot and a final repeat.
#   vLLM:      VLLM_ALIGN=1 (greedy draft, standard rejection, 8192-token budget), recipe running cap.
#   UnieInfra: commit 5e0db812 main checkout only, greedy draft + target-sampled exact match (triton_full),
#              chunk 8192, no coalescing / cache-aware admission / decode-every-prefill, no INT8, util 0.9.
cd /home/smc/workspace/roy
LOCK=logs/GPU_LOCK
M=logs/aligned_sweep.log
: > $M
until ( set -o noclobber; echo "owner=roy-d8(aligned L0 sweep) since=$(date -u +%FT%TZ) release=when 'SWEEP DONE' is written to logs/aligned_sweep.log" > $LOCK ) 2>/dev/null; do sleep 5; done
trap 'rm -f $LOCK' EXIT
until ! pgrep -f "run_round.sh|serve_vllm.sh|serve_unieinfra.sh" >/dev/null; do sleep 2; done
echo "$(date -u +%T) lock taken; unieinfra HEAD $(git -C unieinfra rev-parse --short HEAD)" >> $M

declare -A CAP=([1]=1 [4]=8 [16]=4 [32]=8 [64]=16 [128]=32)   # vLLM recipe running caps (c4: 2x CONC)

cell() {  # conc tag
  local conc=$1 tag=$2 cap=${CAP[$1]}
  echo "$(date -u +%T) start vllm c$conc cap=$cap" >> $M
  VLLM_ALIGN=1 VLLM_MAX_NUM_SEQS=$cap scripts/run_round.sh ${tag}v 900 vllm "$conc"
  local kv
  kv=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" logs/vllm-last-server.log | head -1 | grep -oE "[0-9,]+" | tr -d ,)
  cp logs/vllm-last-server.log logs/vllm-${tag}-c$conc-server.log 2>/dev/null
  echo "$(date -u +%T) done vllm c$conc; KV cache tokens: ${kv:-UNKNOWN}" >> $M
  if [ -z "$kv" ]; then echo "SKIP unieinfra c$conc: no KV size from vLLM" >> $M; return; fi
  echo "$(date -u +%T) start unieinfra c$conc cap=$cap kv=$kv" >> $M
  env -u TOKENSPEED_DSPARK_COUPLED_DRAFT -u TOKENSPEED_CACHE_AWARE_ADMISSION -u TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS \
    ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 \
    MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=$kv GPU_MEMORY_UTILIZATION=0.9 \
    EXTRA_SERVE_ARGS="--sampling-backend triton_full" \
    scripts/run_round.sh ${tag}u 900 unieinfra "$conc"
  echo "$(date -u +%T) done unieinfra c$conc" >> $M
}
for conc in 1 4 32 64 128; do cell $conc L0s$conc; done
cell 16 L0r2            # repeat of the pilot cell, engine order swapped (vLLM first), for a noise estimate
echo "SWEEP DONE $(date -u +%FT%TZ)" >> $M
