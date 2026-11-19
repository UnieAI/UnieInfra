#!/bin/bash
# Mixed-batch cells at the two rows we trail vLLM, single variable vs baseline.
#
# 2026-11-10 10:41Z: the chunk-2048 cell at c64/cap16 died of OOM six minutes into
# its warmup -- "torch.OutOfMemoryError ... Tried to allocate 286.00 MiB. GPU 3 has
# a total capacity of 143.98 GiB of which 0 bytes is free", raised inside a
# *prefill breakable-graph replay* (prefill_graph._replay_narrowed ->
# deepseek_v41._extend_attention -> select_global -> index_topk). The boot log had
# already warned that the reserve priced that ladder at nothing:
#   "prefill:decoder measured 104857600 bytes and priced its 35 unsampled entries
#    at nothing -- every sampled marginal was served from slack"
# so the KV pool was sized as if the prefill graphs were free and the first replay
# workspace that fell outside the capture had nowhere to go. chunk=4096 (every
# baseline cell) does not hit it. Both cells this script runs keep the baseline's
# chunk, coalesce window and sampling backend, so `--enable-mixed-batch` is the
# only difference from the c64-cap16 / c32-cap8 baselines it is compared against.
cd /home/smc/workspace/roy || exit 1
L=logs/mixed_only.log
: > "$L"
echo "MIXED-ONLY start $(date -u +%FT%TZ)" >> "$L"

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

run_cell() {  # tag conc cap extra_env
  local tag=$1 conc=$2 cap=$3 extra=$4 before after
  before=$(ls -1d results/agentx/unieinfra-${tag}-* 2>/dev/null | wc -l)
  wait_free || return 1
  echo "$(date -u +%T) start $tag conc=$conc cap=$cap extra=$extra" >> "$L"
  env CHUNKED_PREFILL_SIZE=4096 MAX_NUM_SEQS=$cap MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
      EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full $extra" \
      scripts/run_round.sh "$tag" 900 unieinfra "$conc" >> "$L" 2>&1
  after=$(ls -1d results/agentx/unieinfra-${tag}-* 2>/dev/null | wc -l)
  if [ "$after" -le "$before" ]; then
    oom=$(grep -l "OutOfMemoryError" logs/unieinfra-server-c64-*.log 2>/dev/null | tail -1)
    echo "$(date -u +%T) CELL FAILED (no export): $tag ${oom:+oom=$oom}" >> "$L"
    return 1
  fi
  echo "$(date -u +%T) done $tag" >> "$L"
}

run_cell mixoff16 64 16 "--enable-mixed-batch" || exit 1
run_cell mixoff8  32  8 "--enable-mixed-batch" || exit 1
echo "MIXED-ONLY DONE $(date -u +%FT%TZ)" >> "$L"
