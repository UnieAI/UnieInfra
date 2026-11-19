#!/bin/bash
# Chunked-prefill size sweep, run once the ladder and mixed-batch queue is done.
#
# Why: at matched concurrency our decode latency tail is the whole deficit -
# c64 cap16 ITL p50/p90 13.7/20.6 ms vs vLLM 18.2 ms p90 (interactivity 49 vs 55),
# c32 cap8 9.1/12.8 vs 12.2. The tail, not the median, is what costs us the
# scoreboard: a 4096-token prefill chunk of a 130K-context prompt stalls every
# decode in the batch for the length of that chunk's DSA pass (~130 ms at the
# measured 31 us/token), and with 8-16 slots arriving continuously those stalls
# land inside the p90 window. Shrinking the chunk trades TTFT (already queue
# dominated: 93 s p90 at c64) for a shorter stall, so measure both ends: 2048
# and 8192 against the 4096 baseline at c64 cap16, plus 2048 at c32 cap8.
#
# Tags carry a chunk prefix so offtable/fair keep these out of the production
# frontier (scripts/offtable.py variant regex).
cd /home/smc/workspace/roy || exit 1
L=logs/chunk_sweep.log
: > "$L"
echo "CHUNK SWEEP start $(date -u +%FT%TZ), waiting for the ladder/mixed queue" >> "$L"
while pgrep -f "[t]est_after_vllm.sh" >/dev/null; do sleep 30; done
echo "ladder/mixed queue finished $(date -u +%FT%TZ)" >> "$L"

export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
for cell in 64:16:chunk2:2048 64:16:chunk8:8192 32:8:chunk2:2048; do
  IFS=: read -r conc cap variant chunk <<< "$cell"
  echo "$(date -u +%T) start conc=$conc cap=$cap chunk=$chunk tag=${variant}off${cap}" >> "$L"
  CHUNKED_PREFILL_SIZE=$chunk MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh ${variant}off${cap} 900 unieinfra "$conc"
  echo "$(date -u +%T) done conc=$conc cap=$cap chunk=$chunk rc=$?" >> "$L"
done
echo "CHUNK SWEEP DONE $(date -u +%FT%TZ)" >> "$L"
