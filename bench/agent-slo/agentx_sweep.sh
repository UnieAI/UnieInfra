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
# The reference cell on GPUs 0-3 (deepseek-DeepSeek-V4.1-Flash-sglang-1) serves
# /models/deepseek-ai/DeepSeek-V4.1-Flash-1017 under the *served* name
# DeepSeek-V4-Flash-Vision-Exp. That alias hid a real difference: the
# DeepSeek-V4-Flash-Vision-Exp checkpoint on disk declares
# architectures=["DeepseekV4ForCausalLM"] (model_type deepseek_v4, 43 layers,
# hidden 4096), while the reference serves the V4.1 checkpoint
# (architectures=["DeepseekV41ForCausalLM"], model_type deepseek_v41). Every
# cell driven before 2026-10-18 measured the V4 vision checkpoint, so those
# numbers are not comparable with the reference cell. The merged tree only
# registers the V41 model, so this is also the only checkpoint it can serve.
MODEL=${MODEL:-/models/deepseek-ai/DeepSeek-V4.1-Flash-1017}
# `/v1/completions` puts the corpus text in front of both engines verbatim;
# `/v1/chat/completions` makes each engine render its own prompt format, and
# those renderings differ in length (sglang/vLLM use a Python encoder for
# DeepSeek-V4.1, a gateway needs a chat template this checkpoint does not
# ship). Same concurrency, same server; only the prompt differs, so a chat cell
# cannot support a TTFT comparison. Every cell in a comparison must be invoked
# with the same ENDPOINT.
ENDPOINT=${ENDPOINT:-completions}
export ENDPOINT
# Model profile. It selects the serving configuration of each arm, because the
# flags that make DeepSeek-V4.1 bootable (engram host tables, EP) mean nothing
# to a dense checkpoint, and a flag carried over from the wrong profile is a
# different configuration, not a faster one.
#   dsv41  - /models/deepseek-ai/DeepSeek-V4.1-Flash-1017, TP=4, EP=TP
#   qwen38 - Qwen3.8-27B, a 27B dense hybrid-attention checkpoint that fits on
#            one H200 at TP=1, so each arm gets a whole device.
PROFILE=${PROFILE:-dsv41}
case "$PROFILE" in
  dsv41) SRV_NAME=DeepSeek-V4-Flash-Vision-Exp ;;
  qwen38) SRV_NAME="$MODEL" ;;
  *) echo "unknown PROFILE $PROFILE (dsv41|qwen38)" >&2; exit 1 ;;
esac
# drive.py counts generated tokens with the checkpoint's own tokenizer when a
# server sends no usage chunk, so token counts mean the same thing on every
# engine instead of counting stream chunks.
export TOKENIZER=${TOKENIZER:-$MODEL}
# Tensor-parallel degree for every kind. The reference cell (sglang, GPUs 0-3)
# runs TP=4, so a cell at a different degree parses prompts and shards experts
# differently and is not a comparable arm.
TP=${TP:-2}
# Overridable so a newer upstream environment image can be tried without
# editing this file: IMG_ENG=... agentx_sweep.sh ours ...
IMG_ENG=${IMG_ENG:-ts-env:gpu-sched39}
IMG_VLL=${IMG_VLL:-vllm/vllm-openai:nightly}
IMG_SGL=${IMG_SGL:-lmsysorg/sglang:dev-dsv41}
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
    case "$PROFILE" in
      qwen38)
        # No tool-call/reasoning parser and no expert parallel: this checkpoint
        # is dense, its own chat template is unused because the cells drive
        # /v1/completions, and TP=1 leaves EP nothing to shard.
        # --enable-mixed-batch: both reference arms pack prefill and decode into
        # one step by default, and this tree's MHA leaf now serves that round.
        # It used to assert it away, which is why every earlier qwen38 cell
        # measured the no-mix configuration -- an arm running a different
        # mechanism than the arms it is compared against.
        RUN_ARGS=(python3 -m tokenspeed.cli serve \
          --model-path "$MODEL" --tp "$TP" --port 8015 --host 0.0.0.0 --trust-remote-code \
          --disable-kvstore \
          --gpu-memory-utilization 0.90 --max-num-seqs 80 --max-model-len 32768 \
          --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
          --block-size 64 \
          --prefill-graph-max-tokens 8192 \
          --enable-mixed-batch)
        ;;
      *)
        RUN_ARGS=(python3 -m tokenspeed.cli serve \
          --model-path "$MODEL" --tp "$TP" --port 8015 --host 0.0.0.0 --trust-remote-code \
          --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
          --disable-kvstore \
          --gpu-memory-utilization 0.87 --max-num-seqs 80 --max-model-len 32768 \
          --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
          --block-size 64 \
          --prefill-graph-max-tokens 8192 \
          --ep-size "$TP")
        ;;
    esac
    # The scheduler already exports what an admission investigation needs:
    # num_requests_running, num_requests_waiting, the KV cache usage ratio and
    # the tokens-per-iteration histogram, updated once per scheduler iteration.
    # Off by default in the engine, so a cell that wants a time series opts in
    # with METRICS_PORT=<host port>; the port is then published and scraped.
    if [ -n "${METRICS_PORT:-}" ]; then
      # --metrics-host is separate from --host on purpose: the orchestrator
      # rewrites the serving --host to loopback inside the container, so a
      # metrics endpoint inheriting it is unreachable from the published port.
      RUN_ARGS+=(--enable-metrics --metrics-port "$METRICS_PORT" --metrics-host 0.0.0.0)
    fi
    # Engram rows go to host memory at every degree, because that is what the
    # reference cell does at TP=4 too: its log reads "engram host table: dropped
    # the page cache of 48 checkpoint files (475 GiB) before pre-faulting the
    # private shard", i.e. it pre-faults a private host shard per rank. Keeping
    # ours on the device instead costs 22.9 GiB of HBM per rank and leaves no
    # room for the KV arena on a 140 GiB H200 (the probe reported -1.13 GiB),
    # so device placement was never the comparable configuration.
    if [ "$PROFILE" != qwen38 ]; then
      RUN_ARGS+=(--engram-host-table --engram-host-table-dir "$HERE/.engram")
    fi
    # --ep-size == --tp: the reference cell runs ep-size==tp-size==4 so the MoE
    # experts shard across the ranks instead of being replicated. This mirrors
    # its layout rather than tuning a knob.
    #
    # TP=2 does not fit this checkpoint at all. With --tp 2 the weight load
    # alone reaches 135.74 GiB per rank on a 140 GiB H200 and the first MoE
    # weight allocation OOMs ("Tried to allocate 2.11 GiB ... 1.38 GiB is
    # free", create_mxfp4_weight_pair). Adding --ep-size 2 did not reduce the
    # footprint, so the 271 GiB total is what TP=2 has to hold per pair of
    # devices. The reference cell therefore runs TP=4, and the only comparable
    # cell here is TP=4 too -- on four free devices, since GPUs 0-3 are the
    # reference.
    # --engram-host-table is not tunable here, it is what makes TP=2 possible.
    # DeepSeek V4.1 carries per-layer Engram n-gram tables in FP8 whose
    # row-sharded per-rank part is 45.8 GiB. With the tables resident on the
    # device, loading them after the 101 GiB of weights OOMs on a 140 GiB H200
    # (measured: "Tried to allocate 45.78 GiB ... 35.90 GiB is free" in
    # RowShardedEngramEmbedding.__init__). The reference cell dodges this by
    # using TP=4, where the row shard is half as wide. Host tables plus UVA
    # gathers are upstream's answer for smaller TP degrees; the host has
    # 1.1 TiB free RAM. The dir matters: the container's /dev/shm is 32 GiB,
    # far below one table.
    mkdir -p "$HERE/.engram"
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
    # Mixed batch is the prefill/decode-overlap mechanism that vLLM/sglang use,
    # and it is OFF by default (ServerArgs.enable_mixed_batch=False), so an arm
    # that wants it must ask. The dsv4 mixed path is correctness-guarded
    # (deepseek_v4.py token-count check), so do NOT disable it in the A/B. A
    # previous baseline replayed the nospec-nomix config
    # (--disable-mixed-batch --prefill-defer-window 32) via extra args.
    INT_PORT=8015
    MN="$MODEL"
    ;;
  vllm)
    RUN_ARGS=(-m vllm.entrypoints.cli.main serve "$MODEL" \
      --tensor-parallel-size "$TP" --max-model-len 32768 \
      --gpu-memory-utilization 0.87 --max-num-seqs 80 --max-num-batched-tokens 8192 \
      --port 8077 --host 0.0.0.0 \
      --served-model-name "$SRV_NAME" --trust-remote-code)
    INT_PORT=8077
    MN="$SRV_NAME"
    ;;
  sglang)
    if [ "$PROFILE" = qwen38 ]; then
      # Plain defaults: no speculative decoding, no hierarchical cache -- nothing
      # the other two arms do not also run. Only a plain cell separates scheduler
      # and kernel work from extra machinery.
      RUN_ARGS=(sglang serve \
        --model-path "$MODEL" --served-model-name "$SRV_NAME" --trust-remote-code \
        --tp-size "$TP" --mem-fraction-static 0.87 \
        --context-length 32768 --chunked-prefill-size 8192 \
        --host 0.0.0.0 --port 8989)
      INT_PORT=8989
      MN="$SRV_NAME"
    else
    # Verbatim the reference cell's command (`docker inspect` on the running
    # container), apart from the port: the arm under test is that configuration,
    # not a re-tuned one. The image is the reference's own too -- the older
    # `sglang-dsv4-vision:pr37257-fix` cannot load this checkpoint
    # ("model type `deepseek_v41` ... Transformers does not recognize").
    RUN_ARGS=(sglang serve \
      --model-path "$MODEL" \
      --served-model-name "$SRV_NAME" --trust-remote-code \
      --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41 \
      --default-chat-template-kwargs '{"thinking": true}' \
      --enable-cache-report --enable-hierarchical-cache --hicache-ratio 2.0 \
      --hicache-write-policy write_through --hicache-io-backend kernel \
      --enable-metrics --tp-size "$TP" --ep-size "$TP" --attention-backend dsv4 \
      --moe-runner-backend flashinfer_mxfp4 \
      --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
      --enable-decoder-swa-bounded-replay --cuda-graph-max-bs-decode 64 \
      --mem-fraction-static 0.8 --context-length 262144 \
      --chunked-prefill-size 4096 \
      --enable-multimodal --mm-attention-backend fa3 \
      --host 0.0.0.0 --port 8989 --watchdog-timeout 3600)
    INT_PORT=8989
    MN="$SRV_NAME"
    fi
    ;;
  *) echo "unknown kind $KIND (ours|vllm|sglang)" >&2; exit 1 ;;
esac

# Forward extras passed by the caller (e.g. a different max-num-seqs).
if [ "$#" -gt 0 ]; then
  RUN_ARGS+=("$@")
fi

case "$KIND" in
  ours)
    # The mount below supplies both the Python bindings *and* the compiled
    # extension (the .so lives inside the package directory), so it overrides
    # whatever version the image ships. That is deliberate -- the tree stays the
    # source of truth -- but it also means a stale .so silently runs older C++
    # scheduling semantics underneath newer Python. It was stale by three days
    # when this check was added. Fail fast instead of measuring the wrong C++.
    SCHED_PKG=$HERE/UnieInfra/tokenspeed-scheduler/python/tokenspeed_scheduler
    SCHED_SO=$(ls "$SCHED_PKG"/tokenspeed_scheduler_ext*.so 2>/dev/null | head -1)
    REBUILD_HINT="docker run --rm -v /workspace:/workspace -w /workspace/roy/UnieInfra/tokenspeed-scheduler ts-env:gpu pip wheel . -w /workspace/roy/.wheels --no-deps, then unzip the .so over $SCHED_PKG"
    if [ -z "$SCHED_SO" ]; then
      echo "no tokenspeed_scheduler_ext*.so in $SCHED_PKG: the mount would hide the image's copy and the engine cannot import the scheduler." >&2
      echo "Rebuild: $REBUILD_HINT" >&2
      exit 1
    fi
    newer=$(find "$HERE/UnieInfra/tokenspeed-scheduler/csrc" "$HERE/UnieInfra/tokenspeed-scheduler/bindings" -newer "$SCHED_SO" -type f 2>/dev/null | wc -l)
    if [ "$newer" -ne 0 ]; then
      echo "$newer scheduler source files are newer than $(basename "$SCHED_SO"), so this cell would run the old C++ scheduler under the new Python." >&2
      echo "Rebuild: $REBUILD_HINT" >&2
      exit 1
    fi
    # The package directory lives in different places depending on how the image
    # was built (user install vs dist-packages), and a mount over the wrong path
    # silently does nothing -- the engine then imports whatever the image ships,
    # which is exactly the staleness this block exists to prevent. Ask the image.
    SCHED_SITE=$(docker run --rm --entrypoint python3 "$IMG_ENG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
    if [ -z "$SCHED_SITE" ]; then
      echo "cannot determine site-packages inside $IMG_ENG" >&2
      exit 1
    fi
    PUBLISH_ARGS=(-p "$PORT:$INT_PORT")
    if [ -n "${METRICS_PORT:-}" ]; then
      PUBLISH_ARGS+=(-p "$METRICS_PORT:$METRICS_PORT")
    fi
    docker run -d --name "$NAME" --gpus \"device=$GPU\" --shm-size 32g \
      "${PUBLISH_ARGS[@]}" -v /models:/models -v /workspace:/workspace \
      -v "$HERE/UnieInfra/tokenspeed-scheduler/python/tokenspeed_scheduler:$SCHED_SITE/tokenspeed_scheduler:ro" \
      -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
      -e FLASHINFER_WORKSPACE_BASE=/workspace/roy/.fiws \
      -e PYTHONPATH=/workspace/roy/UnieInfra/python:/workspace/roy/UnieInfra/tokenspeed-kernel/python \
      -e TOKENSPEED_DSV4_FP8_INDEXER_FUSED_VERIFY="${DSV4_FP8_VERIFY:-0}" \
      -e TOKENSPEED_DSV4_FP8_INDEXER_FUSED="${DSV4_FP8_FUSED:-0}" \
      -e TORCH_EXTENSIONS_DIR="${TORCH_EXT_DIR:-/workspace/roy/.torch_ext}" \
      -e TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR=/workspace/roy/.autotune \
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
    # Engram host tables are opt-in in this image (SGLANG_ENABLE_DSV41_ENGRAM_
    # HOST_TABLE defaults to False), and the reference cell sets them: without
    # them the weights leave no room for the KV cache at --mem-fraction-static
    # 0.8 ("minimum viable = 0.8838"). The reference container's own environment
    # carries both of these, so the arm inherits them verbatim.
    SGL_ENV=()
    if [ "$PROFILE" = dsv41 ]; then
      SGL_ENV+=(-e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 -e SGLANG_DEFAULT_THINKING=1)
    fi
    docker run -d --name "$NAME" --gpus \"device=$GPU\" --shm-size 32g \
      -p "$PORT:$INT_PORT" -v /models:/models -v /workspace:/workspace \
      -v "$HERE/.sglang_cache:/root/.cache" "${SGL_ENV[@]}" \
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
