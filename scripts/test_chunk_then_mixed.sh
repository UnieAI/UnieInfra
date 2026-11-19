#!/bin/bash
# Chunk-size sweep, then the mixed-batch cells, with a VRAM pre-flight.
#
# 2026-11-10 10:11Z: the first chunk sweep lost BOTH cells without measuring
# anything. When the previous (mixed) queue was interrupted, its four
# tokenspeed::scheduler_tp* processes survived the kill in state R, spinning and
# holding ~78 GB of VRAM. run_round.sh then launched a server that died in load
# with torch.OutOfMemoryError ("GPU 1 has 0 bytes free") and the sweep happily
# advanced to the next cell (rc=0 from run_round_*  because the failure is in the
# server log, not the bench). This script therefore refuses to start a cell until
# the device is actually free, and stops the whole sweep if a cell produces no
# export instead of burning the next 20 minutes on the same OOM.
#
# Cells: c64 cap16 chunk 2048 / 8192 (the tail-interference test at the row we
# trail vLLM by 0.84x), c32 cap8 chunk 2048, then mixed 32:8 and 64:16.
cd /home/smc/workspace/roy || exit 1
L=logs/chunk_then_mixed.log
: > "$L"
echo "CHUNK+MIXED start $(date -u +%FT%TZ)" >> "$L"

vram_used_gb() {
  docker exec -u root ts-dev rocm-smi --showmeminfo vram 2>/dev/null \
    | awk '/Used Memory \(B\)/ {s += $NF} END {printf "%d", s/1024/1024/1024}'
}

wait_free() {
  for _ in $(seq 1 40); do
    u=$(vram_used_gb)
    [ -n "$u" ] && [ "$u" -lt 8 ] && echo "$(date -u +%T) vram free (${u} GB) OK" >> "$L" && return 0
    echo "$(date -u +%T) waiting for vram, used=${u} GB" >> "$L"
    sleep 15
  done
  echo "$(date -u +%T) ABORT: vram still busy, used=$(vram_used_gb) GB" >> "$L"
  return 1
}

run_cell() {  # tag conc cap chunk extra_env
  local tag=$1 conc=$2 cap=$3 chunk=$4 extra=$5 before after
  before=$(ls -1d results/agentx/unieinfra-${tag}-* 2>/dev/null | wc -l)
  wait_free || return 1
  echo "$(date -u +%T) start $tag conc=$conc cap=$cap chunk=$chunk" >> "$L"
  env CHUNKED_PREFILL_SIZE=$chunk MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
      EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full $extra" \
      scripts/run_round.sh "$tag" 900 unieinfra "$conc" >> "$L" 2>&1
  after=$(ls -1d results/agentx/unieinfra-${tag}-* 2>/dev/null | wc -l)
  if [ "$after" -le "$before" ]; then
    echo "$(date -u +%T) CELL FAILED (no export): $tag - see logs/round-${tag}.log" >> "$L"
    return 1
  fi
  echo "$(date -u +%T) done $tag" >> "$L"
}

CP='export TOKENSPEED_IRIS_Q8_ALLREDUCE=0; '
run_cell chunk2off16 64 16 2048 ""                       || exit 1
run_cell chunk8off16 64 16 8192 ""                       || exit 1
run_cell chunk2off8  32  8 2048 ""                       || exit 1
run_cell mixoff8     32  8 16384 "--enable-mixed-batch"  || exit 1
run_cell mixoff16    64 16 16384 "--enable-mixed-batch"  || exit 1
echo "CHUNK+MIXED DONE $(date -u +%FT%TZ)" >> "$L"
