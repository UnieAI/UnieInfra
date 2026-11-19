#!/bin/bash
# Mixed prefill/decode batching in OFFICIAL mode, at the two cells we lose.
#
# Entry 43/44 tested --enable-mixed-batch in stress mode on 2026-11-06 and found
# mixed wins at c4-c16 but loses at c64 (96.2 vs 107.3 RPM), with the 4096-token
# reference chunk starving prefill (TTFT p50 0.7 -> 29 s). Both cells predate
# cache-aware admission, per-chunk admission and the coupled drafter, and neither
# ran in official mode, so the conclusion is stale. The mid-range is also exactly
# where we trail vLLM (c16 0.97x, c32 0.93x, c64 0.84x), and mixed batching is the
# one scheduling-level knob that can move both TTFT and ITL together: decodes ride
# in prefill rounds, so prefill latency stops being a queue of its own.
#
# Large mixed chunks (entry 44's follow-up). The chunk budget is CHUNKED_PREFILL_SIZE:
# the TOKENSPEED_MIXED_REF_CHUNK knob entry 44 used no longer exists in this tree
# (server_args picks 4096/8192 from the architecture), so the prefill side of a mixed
# round is the chunked-prefill size and nothing else.
cd /home/smc/workspace/roy || exit 1
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
export TOKENSPEED_IRIS_Q8_ALLREDUCE=0
export CHUNKED_PREFILL_SIZE=16384
L=logs/mixed_official.log
: > "$L"
echo "MIXED OFFICIAL start $(date -u +%FT%TZ)" >> "$L"
for cell in 32:8 64:16; do
  conc=${cell%%:*}; cap=${cell##*:}
  echo "$(date -u +%T) start conc=$conc cap=$cap" >> "$L"
  MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
    EXTRA_SERVE_ARGS="--enable-mixed-batch --prefill-coalesce-ms 100 --sampling-backend triton_full" \
    scripts/run_round.sh mixoff${cap} 900 unieinfra "$conc"
  echo "$(date -u +%T) done conc=$conc cap=$cap rc=$?" >> "$L"
done
echo "MIXED OFFICIAL DONE $(date -u +%FT%TZ)" >> "$L"
