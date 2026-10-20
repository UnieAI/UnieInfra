#!/usr/bin/env bash
# Start the UnieInfra (tokenspeed-fork) OpenAI server for benchmarking on sm_120.
# Runs inside the Blackwell dev container `unieinfra-dev`.
set -euo pipefail
MODEL="${MODEL:?set MODEL=/path/to/model}"
PORT="${PORT:-8000}"
GPU="${GPU:-1}"
CONTAINER="${CONTAINER:-unieinfra-dev}"
DTYPE="${DTYPE:-bfloat16}"; GMU="${GMU:-0.85}"; MAXLEN="${MAXLEN:-4096}"; ATTN="${ATTN:-triton}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"; MAX_CGRAPH="${MAX_CGRAPH:-}"   # optional high-concurrency tuning
DISABLE_RADIX="${DISABLE_RADIX:-0}"   # keep prefix cache ON (real feature; production has it). Clean separation comes from FLUSH between levels, not from disabling it.
# Server-side Prometheus counters (tokenspeed:generation_tokens, ...).
#
# DEFAULT IS OFF, deliberately: turning them on changes what the engine does per
# token (the frontend updates TTFT/ITL histograms in the output processor), so
# flipping it silently would invalidate measurements already in flight.
#
# Note the asymmetry it papers over: vLLM records the equivalent
# vllm:generation_tokens_total by default (it needs --disable-log-stats to stop),
# so an ENABLE_METRICS=0 TokenSpeed arm is NOT paying a cost the vLLM arm pays.
# Run at least one paired A/B with ENABLE_METRICS=1 before quoting a delta.
#
# ENABLE_METRICS=1 makes /metrics available at http://<host>:$((PORT+1))/metrics
# -- the control sidecar port, NOT the gateway port. The gateway has no /metrics.
ENABLE_METRICS="${ENABLE_METRICS:-0}"
# Free-form passthrough for any `tokenspeed serve` flag that has no dedicated
# knob here yet (e.g. --disable-kvstore, --enable-mixed-batch). Appended last so
# it can also override an earlier flag. Recorded in launch.json for provenance.
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA=""
[ -n "$MAX_NUM_SEQS" ] && EXTRA="$EXTRA --max-num-seqs $MAX_NUM_SEQS"
[ -n "$MAX_CGRAPH" ] && EXTRA="$EXTRA --max-cudagraph-capture-size $MAX_CGRAPH"
# WARNING: `--disable-radix-cache` does not exist in this TokenSpeed tree. The
# flag was renamed; the working opt-out is `--no-enable-prefix-caching`
# (python/tokenspeed/runtime/utils/server_args.py:1638-1650). Unrecognised flags
# fall through to the smg gateway (cli/_argsplit.py rule 7), whose clap parser
# rejects them, so DISABLE_RADIX=1 kills the gateway at startup and has never
# produced a prefix-cache-off run. Left as-is (unused at the default 0) rather
# than fixed silently; see the note in the serving-gap report.
# Caveat even once renamed: --no-enable-prefix-caching only stops prefix
# MATCHING. The radix tree is still built and the host KVStore still allocated
# (kv_prefix_cache.cpp:163-165; event_loop.py:449-453).
[ "$DISABLE_RADIX" = "1" ] && EXTRA="$EXTRA --disable-radix-cache"
[ "$ENABLE_METRICS" = "1" ] && EXTRA="$EXTRA --enable-metrics"
[ -n "$EXTRA_ARGS" ] && EXTRA="$EXTRA $EXTRA_ARGS"

# --- record launch config (card, count, params) for reproducible result saving ---
NUM_GPUS=$(echo "$GPU" | tr ',' '\n' | grep -c .)
GPU_MODEL=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
RESDIR="$(cd "$(dirname "$0")/.." && pwd)/results"; mkdir -p "$RESDIR"
cat > "$RESDIR/unieinfra.launch.json" <<JSON
{"engine":"unieinfra","model":"$MODEL","served_model_name":"bench-model","port":$PORT,
 "gpu_ids":"$GPU","num_gpus":$NUM_GPUS,"gpu_model":"$GPU_MODEL",
 "dtype":"$DTYPE","gpu_memory_utilization":$GMU,"max_model_len":$MAXLEN,
 "attention_backend":"$ATTN","tp_size":$NUM_GPUS,"container":"$CONTAINER",
 "max_num_seqs":"$MAX_NUM_SEQS","max_cudagraph_capture_size":"$MAX_CGRAPH",
 "extra_args":"$EXTRA_ARGS"}
JSON
echo "[launch] unieinfra on $NUM_GPUS x $GPU_MODEL (gpu ids=$GPU); config -> $RESDIR/unieinfra.launch.json"

docker exec "$CONTAINER" bash -lc "
  cd /workspace/roy/UnieInfra && source benchmark/scripts/sm120_env.sh
  export CUDA_VISIBLE_DEVICES=$GPU
  tokenspeed serve '$MODEL' \
    --served-model-name bench-model \
    --host 0.0.0.0 --port $PORT \
    --dtype $DTYPE --gpu-memory-utilization $GMU --max-model-len $MAXLEN \
    --attention-backend $ATTN --trust-remote-code $EXTRA
"
