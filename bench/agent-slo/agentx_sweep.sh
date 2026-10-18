#!/bin/bash
# No-replay concurrency sweep for the AgentX SLO benchmark.
#
# Boots a FRESH serving cell (docker container on a chosen GPU set), runs
# drive.py at each requested concurrency against it, saves one raw JSON per
# (engine, concurrency), tears the cell down. Every cell starts with an empty
# prefix cache so the run measures reuse-under-load, not cache replay of the
# measured corpus. Warmup uses a disjoint corpus (drive.py).
#
# Usage:
#   agentx_sweep.sh ours|vllm|sglang GPU_SET PORT TAG CONCS NREQ DATA WARM
#     GPU_SET : e.g. "6,7"  (docker device indices on the host)
#     PORT    : host HTTP port to publish (unused externally by the host python)
#     TAG     : short run tag used as container name + output prefix
#     CONCS   : comma-separated concurrency list, e.g. "4,8,16"
#     NREQ    : requests per cell
#     DATA    : measured corpus (.jsonl, one {"prompt":...} per line)
#     WARM    : warmup corpus (must NOT share prefix with DATA)
#   extra args after DATA/WARM are appended to the engine serve command.
set -u
KIND=${1:?kind}; shift
GPU=${1:?gpu_set}; shift
PORT=${1:?port}; shift
TAG=${1:?tag}; shift
CONCS=${1:?concs}; shift
NREQ=${1:?nreq}; shift
DATA=${1:?data}; shift
WARM=${1:?warm}; shift

HERE=/workspace/roy
MODEL=/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
IMG_ENG=ts-env:gpu-sched39
IMG_VLL=vllm/vllm-openai:nightly
IMG_SGL=sglang-dsv4-vision:pr37257-fix
NAME="agentx-${KIND}-${TAG}"
OUTDIR=$HERE/ab_real/agentx-sweep
mkdir -p "$HERE/logs" "$OUTDIR"

# Leftover temp file from an earlier debug hack; keep the working tree clean-ish.
: "${KEEP:=0}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

ENV_ARGS=()
RUN_ARGS=()

case "$KIND" in
  ours)
    RUN_ARGS=(python3 -m tokenspeed.cli serve \
      --model-path "$MODEL" --tp 2 --port 8015 --host 0.0.0.0 --trust-remote-code \
      --reasoning-parser deepseek_v31 \
      --disable-kvstore \
      --gpu-memory-utilization 0.87 --max-num-seqs 80 --max-model-len 32768 \
      --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
      --block-size 64 \
      --prefill-graph-max-tokens 8192)
    # No --admission-reserve here. Upstream (merged 2026-10-18) deleted that
    # flag: it was a knob over a quantity the admission code computed wrongly.
    # The merged scheduler reserves per cache group from each request's own
    # demand (ReservePrefillDemands in
    # tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp), so there is
    # no global mode to select. Passing the old flag fails to boot
    # ("unrecognized arguments"); the audit for it lives in
    # unieinfra/xref/72-PARITY-STRATEGY.md §2 instead of an A/B.
    # Do NOT pass --prefill-graph-capture-sizes here. That flag replaces the
    # prefill token-bucket ladder with an explicit list, and a single entry
    # (e.g. "8192") collapses it to one rung -- so every prefill step replays
    # an 8192-token graph however few tokens it really holds. Measured on
    # AgentX c16: a 127-token prefill cost 363 ms of forward instead of 157 ms,
    # and the cell ran 127.5 s instead of 92.3 s. The default ladder is dense
    # (56 rungs, <=1.01x padding) and cheap, because all captures share one
    # stream and mempool. It was forced to a single rung only because
    # weak_ref_tensor could not compile in a container with a read-only
    # $HOME, which pins graph-pool blocks and makes capture memory scale with
    # the bucket sum; that fallback is fixed in
    # tokenspeed_kernel/ops/transform/weak_ref.py.
    # Do NOT pass --max-cudagraph-capture-size either. Capping it below
    # --max-num-seqs does not save memory, it removes the decode graph: every
    # decode step wider than the cap falls back to the eager path, and nothing
    # reports it. Capped at 40 with max_num_seqs=80, AgentX c64 ran 124.5 s at
    # 262 total tok/s and 229 ms TPOT; with the engine default (160, clamped to
    # max_num_seqs=80) the same cell ran 59.8 s at 547 tok/s and 82 ms TPOT,
    # with graph=True at bs=59/64/69/71/73. get_batch_sizes_to_capture() now
    # warns when the cap is the binding constraint.
    # Mixed batch is ON by default (ServerArgs.enable_mixed_batch=True) and is
    # the prefill/decode-overlap mechanism that vLLM/sglang use; the dsv4 mixed
    # path is correctness-guarded (deepseek_v4.py token-count check), so do NOT
    # disable it in the A/B. A previous baseline replayed the nospec-nomix
    # config (--disable-mixed-batch --prefill-defer-window 32) via extra args.
    INT_PORT=8015
    MN="$MODEL"
    ;;
  vllm)
    RUN_ARGS=(-m vllm.entrypoints.cli.main serve "$MODEL" \
      --tensor-parallel-size 2 --max-model-len 32768 \
      --gpu-memory-utilization 0.87 --max-num-seqs 80 --max-num-batched-tokens 8192 \
      --port 8077 --host 0.0.0.0 \
      --served-model-name DeepSeek-V4-Flash-Vision-Exp --trust-remote-code)
    INT_PORT=8077
    MN=DeepSeek-V4-Flash-Vision-Exp
    ;;
  sglang)
    RUN_ARGS=(sglang serve \
      --model-path "$MODEL" \
      --served-model-name DeepSeek-V4-Flash-Vision-Exp --trust-remote-code \
      --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
      --default-chat-template-kwargs '{"thinking": true}' \
      --enable-cache-report --enable-hierarchical-cache --hicache-ratio 2.0 \
      --hicache-write-policy write_through --hicache-io-backend kernel \
      --enable-metrics --tp 2 --kv-cache-dtype auto \
      --moe-runner-backend flashinfer_mxfp4 \
      --speculative-algorithm DSPARK --mem-fraction-static 0.82 \
      --context-length 131072 --chunked-prefill-size 4096 \
      --swa-full-tokens-ratio 0.1 --skip-server-warmup \
      --enable-multimodal --mm-attention-backend fa3 --mm-enable-dp-encoder \
      --enable-mm-global-cache --mm-processor-worker-num 8 --mm-io-worker-num 8 \
      --host 0.0.0.0 --port 8989)
    INT_PORT=8989
    MN=DeepSeek-V4-Flash-Vision-Exp
    ;;
  *) echo "unknown kind $KIND (ours|vllm|sglang)" >&2; exit 1 ;;
esac

# Forward extras passed by the caller (e.g. a different max-num-seqs).
if [ "$#" -gt 0 ]; then
  RUN_ARGS+=("$@")
fi

case "$KIND" in
  ours)
    docker run -d --name "$NAME" --gpus \"device=$GPU\" --shm-size 32g \
      -p "$PORT:$INT_PORT" -v /models:/models -v /workspace:/workspace \
      -v "$HERE/UnieInfra/tokenspeed-scheduler/python/tokenspeed_scheduler:/home/runner/.local/lib/python3.12/site-packages/tokenspeed_scheduler:ro" \
      -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
      -e FLASHINFER_WORKSPACE_BASE=/workspace/roy/.fiws \
      -e PYTHONPATH=/workspace/roy/UnieInfra/python:/workspace/roy/UnieInfra/tokenspeed-kernel/python \
      -e TOKENSPEED_DSV4_FP8_INDEXER_FUSED_VERIFY="${DSV4_FP8_VERIFY:-0}" \
      -e TOKENSPEED_DSV4_FP8_INDEXER_FUSED="${DSV4_FP8_FUSED:-0}" \
      -e TOKENSPEED_SCHED_LOOP_TIMING="${SCHED_TIMING:-0}" \
      -e TORCH_EXTENSIONS_DIR=/workspace/roy/.torch_ext \
      "$IMG_ENG" bash -lc "${RUN_ARGS[*]}"
    ;;
  vllm)
    mkdir -p "$HERE/.vllm_cache"
    docker run -d --name "$NAME" --gpus \"device=$GPU\" --shm-size 32g \
      --entrypoint python3 \
      -p "$PORT:$INT_PORT" -v /models:/models -v /workspace:/workspace \
      -v "$HERE/.vllm_cache:/root/.cache" \
      -e NCCL_NVLS_ENABLE=0 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
      "$IMG_VLL" "${RUN_ARGS[@]}"
    ;;
  sglang)
    mkdir -p "$HERE/.sglang_cache"
    docker run -d --name "$NAME" --gpus \"device=$GPU\" --shm-size 32g \
      -p "$PORT:$INT_PORT" -v /models:/models -v /workspace:/workspace \
      -v "$HERE/.sglang_cache:/root/.cache" \
      "$IMG_SGL" "${RUN_ARGS[@]}"
    ;;
esac

# The readiness probe below only proves that *something* answers on $PORT. A
# shared server already bound there answers 200 before this cell has loaded a
# single weight, and the run then silently measures somebody else's model. That
# is not hypothetical: host port 8989 is held by a shared multi-model sglang
# gateway, and a `docker run` that failed with "address already in use" was
# still reported as "UP after 1s" and driven. Require the container we just
# launched to actually be running before trusting the probe.
st=missing
for _ in $(seq 1 10); do
  st=$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo missing)
  [ "$st" = "true" ] && break
  sleep 1
done
if [ "$st" != "true" ]; then
  echo "[$TAG] FAILED: container $NAME is not running -- docker run did not start it."
  echo "[$TAG] Hint: if the error was 'address already in use', pick another PORT."
  docker logs "$NAME" 2>&1 | tail -20 | sed 's/^/  /'
  exit 1
fi

LOG=$HERE/logs/agentx-sweep-$TAG.log
: > "$LOG"

echo "[$TAG] launched kind=$KIND gpu=$GPU port=$PORT name=$NAME"
docker logs -f "$NAME" > "$LOG" 2>&1 &
LOGPID=$!

up=0
for i in $(seq 1 900); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then up=1; echo "[$TAG] UP after ${i}s"; break; fi
  if [ $((i % 30)) -eq 0 ]; then echo "[$TAG] waiting ${i}s (code=$code)"; tail -5 "$LOG" | sed 's/^/  /'; fi
  sleep 2
done
if [ "$up" -ne 1 ]; then echo "[$TAG] FAILED to come up"; tail -120 "$LOG"; kill "$LOGPID" 2>/dev/null; exit 1; fi

for CONC in $(echo "$CONCS" | tr ',' ' '); do
  OUT=$OUTDIR/agentx_${KIND}_${TAG}_c${CONC}.json
  echo "[$TAG] conc=$CONC -> $OUT"
  if ! "$HERE/UnieInfra/.venv/bin/python" "$HERE/UnieInfra/bench/agent-slo/drive.py" \
       "http://127.0.0.1:$PORT" "$MN" "$DATA" "$CONC" "$NREQ" "$OUT" "$WARM"; then
    echo "[$TAG] run failed at conc=$CONC"; kill "$LOGPID" 2>/dev/null; exit 1
  fi
done

kill "$LOGPID" 2>/dev/null
if [ "$KEEP" = "1" ]; then
  echo "[$TAG] KEEP=1 -> leaving container $NAME up; test: curl http://127.0.0.1:$PORT/v1/models"
else
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "[$TAG] done; cell torn down"
fi
