#!/bin/bash
# Decode-only A/B at matched batch (16 concurrent users) and matched context.
#
# Why: the c64-cap16 AgentX row is where we trail vLLM (60.3 RPM @ inter 49 vs
# 71.5 @ 55) and every scheduling lever tried so far (chunk, caps, coalescing)
# moved less than the noise. Two explanations remain and they call for opposite
# work:
#   (a) our decode *step* is slower than vLLM's at the same batch  -> kernel work;
#   (b) our step is fine and the loss is prefill interference/stalls -> scheduler.
# AgentX cannot separate them because its ITL is measured across prefill rounds
# and a growing context. bench_decode.py holds both engines at n=16 fixed prompts,
# identical decode length, ignore_eos, and reports ms/token per user directly
# comparable with our ITL p50.
#
# Both engines run the same checkpoint, TP4, bf16, one server at a time. Contexts
# 8k and 30k bracket the AgentX turn-1..turn-N range.
cd /home/smc/workspace/roy || exit 1
L=logs/decode_ab.log
: > "$L"
echo "DECODE-AB start $(date -u +%FT%TZ)" >> "$L"

vram_used_gb() {
  docker exec -u root ts-dev rocm-smi --showmeminfo vram 2>/dev/null \
    | awk '/Used Memory \(B\)/ {s += $NF} END {printf "%d", s/1024/1024/1024}'
}

wait_free() {
  for _ in $(seq 1 60); do
    u=$(vram_used_gb)
    [ -n "$u" ] && [ "$u" -lt 8 ] && echo "$(date -u +%T) vram free (${u} GB)" >> "$L" && return 0
    echo "$(date -u +%T) waiting for vram, used=${u} GB" >> "$L"
    sleep 15
  done
  echo "$(date -u +%T) ABORT: vram busy, used=$(vram_used_gb) GB" >> "$L"
  return 1
}

bench() {  # engine label tokens
  local tag=$1 label=$2 tokens=$3
  echo "$(date -u +%T) $label prompt~$tokens" >> "$L"
  python3 scripts/bench_decode.py --n 16 --tokens "$tokens" --max-tokens 256 2>&1 | tail -2 | tee -a "$L"
}

# Kernel breakdown of one decode window at n=16. The observable endpoints are the
# fork's own (/start_profile, /stop_profile) -> a torch chrome trace that
# scripts/kernel_breakdown.py summarises. Timing runs finish first so this
# profiled window is never part of a reported number.
profile_decode16() {
  echo "$(date -u +%T) profiling 4s of decode at n=16" >> "$L"
  python3 scripts/bench_decode.py --n 16 --tokens 8000 --max-tokens 192 \
      > logs/prof16-decode.log 2>&1 &
  local pid=$!
  sleep 25
  curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' \
      -d '{"output_dir": "/logs/prof-decode16", "activities": ["GPU"]}' >> "$L" 2>&1; echo >> "$L"
  sleep 4
  curl -s -X POST localhost:8000/stop_profile >> "$L" 2>&1; echo >> "$L"
  wait $pid
  sleep 15
  local trace
  trace=$(ls -t logs/prof-decode16/*.json.gz 2>/dev/null | head -1)
  if [ -n "$trace" ]; then
    echo "--- kernel breakdown: $trace" >> "$L"
    python3 scripts/kernel_breakdown.py "$trace" 25 2>&1 | tee -a "$L"
  else
    echo "no trace in logs/prof-decode16" >> "$L"
  fi
}

# --- UnieInfra: the c64-cap16 server, unchanged --------------------------------
# Launched exactly the way the official sweep launches it (MAX_NUM_SEQS only), so
# the per-step number explains the reported cell. Passing extra serve flags here
# (e.g. --prefill-coalesce-ms, whose default is 0.0) would profile a different
# server than the one the row in the table came from.
scripts/stop_servers.sh >> "$L" 2>&1
wait_free || exit 1
echo "$(date -u +%T) start unieinfra c64 cap16 (official sweep config)" >> "$L"
env MAX_NUM_SEQS=16 scripts/serve_unieinfra.sh 64 >> "$L" 2>&1 \
    || { echo "serve failed" >> "$L"; exit 1; }
if ! scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> "$L" 2>&1; then
  echo "$(date -u +%T) unieinfra FAILED to become ready" >> "$L"
else
  bench unieinfra unieinfra-c64cap16 8000
  bench unieinfra unieinfra-c64cap16 30000
  profile_decode16
fi
scripts/stop_servers.sh >> "$L" 2>&1

# --- vLLM: the reference recipe at c64 (its max-num-seqs is CONC/4 = 16) ------
wait_free || exit 1
echo "$(date -u +%T) start vllm c64 cap16" >> "$L"
scripts/serve_vllm.sh 64 >> "$L" 2>&1 || { echo "vllm serve failed" >> "$L"; exit 1; }
if ! scripts/wait_ready.sh 3000 "" >> "$L" 2>&1; then
  echo "$(date -u +%T) vllm FAILED to become ready" >> "$L"
else
  bench vllm vllm-c64cap16 8000
  bench vllm vllm-c64cap16 30000
fi
scripts/stop_servers.sh >> "$L" 2>&1
echo "DECODE-AB DONE $(date -u +%FT%TZ)" >> "$L"
