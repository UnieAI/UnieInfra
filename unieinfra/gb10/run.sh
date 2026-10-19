#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 UnieAI
#
# Bring-up test of TokenSpeed on the GB10. Run on the GB10 host after sync.sh:
#
#   ~/unieinfra-main/src/unieinfra/gb10/run.sh prep     # pull images, download the model
#   ~/unieinfra-main/src/unieinfra/gb10/run.sh check    # level 1: kernels and kernel tests
#   ~/unieinfra-main/src/unieinfra/gb10/run.sh window   # level 2: serve, smoke test, benchmark
#   ~/unieinfra-main/src/unieinfra/gb10/run.sh restore  # restart the resident service by hand
#
# The image comes from prebuilt wheels when sync.sh shipped some (built on an
# arm64 workstation by build_wheels.sh), and is otherwise built here from the
# Dockerfile -- which needs the memory the resident service normally holds, so
# that path refuses to start unless enough is free.
#
# CPU and GPU share 128 GB on GB10, and the kernel OOM killer would pick the
# resident service first (it has the highest oom_score on the box). `check`
# therefore runs in memory-capped containers under a watchdog that kills them
# if the host runs low.
#
# `window` stops the service if it is running and then restarts it on every exit
# path, including a failed step and Ctrl-C; a service already stopped by hand is
# left stopped. Not covered: a host reboot mid-window, after which `docker start`
# it by hand (unless-stopped skips a container stopped on purpose). The test
# server binds 127.0.0.1 on a port the service does not use: the host runs a
# cloudflared tunnel that would publish anything on the service's port.
set -euo pipefail

SERVICE="${GB10_SERVICE:-DeepSeek-V4-Flash-Vision-Exp-EXL3}"
WORK="${GB10_WORK:-$HOME/unieinfra-main}"
MODEL_REPO="${GB10_MODEL_REPO:-Qwen/Qwen3.8-27B}"
MODEL_DIR="${GB10_MODEL_DIR:-$HOME/models/$MODEL_REPO}"
BASE_IMAGE="lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.7.1rc2"
# Load generator for level 2; only its `vllm bench serve` client is used.
BENCH_IMAGE="${GB10_BENCH_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
# Not `unieinfra:gb10-main`: the GB10 CI runs its tasks in that image, so a
# manual build under that name would change what CI tests.
IMAGE="unieinfra:gb10-dev"
CTR="unieinfra-dev-test"
INSTALL_CTR="unieinfra-dev-install"
PORT="${GB10_PORT:-18080}"
# Fraction of the unified 128 GB the engine may plan for. Lower than the 0.88
# used on discrete cards: here the same pool also backs the host processes and
# the page cache the weights are read through.
MEM_UTIL="${GB10_MEM_UTIL:-0.6}"
CONCURRENCY="${GB10_CONCURRENCY:-1 4 16 64}"
# Prompts per benchmark point: 4 per concurrent slot, at least this many. A
# 27B dense decode is ~5 tok/s per stream here, so c=1 sets the run time.
BENCH_MIN_PROMPTS="${GB10_BENCH_MIN_PROMPTS:-8}"
# Engine flags for level 2: the GB10 CI model-test flags (test/ci/eval/*.yaml)
# with the two changes GB10 needs to start at all.
# - No `--kv-cache-dtype fp8_e4m3`: an fp8 cache makes the runtime cast q to
#   fp8, and no mha_extend_with_kvcache kernel on sm_12x accepts fp8 q.
# - `--disable-kvstore --retraction-snapshot-ratio 0`: both host tiers are on by
#   default and sized off the device cache (103 GB for Qwen3.8-27B), but on GB10
#   host memory is the same memory the device cache already holds.
SERVE_ARGS="${GB10_SERVE_ARGS:---trust-remote-code --attention-backend triton --moe-backend triton --chunked-prefill-size 8192 --max-model-len 8192 --max-num-seqs 64 --disable-kvstore --retraction-snapshot-ratio 0}"
# `check` only: container memory cap, and the host MemAvailable floor below
# which the watchdog kills the test.
CHECK_MEM_GB="${GB10_CHECK_MEM_GB:-5}"
GUARD_MB="${GB10_GUARD_MB:-1536}"
# Building the image here runs nvcc on 16 jobs.
BUILD_MIN_FREE_GB=48

RUNTIME_ENV=(
  -e HOME=/root
  -e TRITON_CACHE_DIR=/root/.cache/triton
  -e CUDA_CACHE_PATH=/root/.cache/nv
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a
  -e TORCH_CUDA_ARCH_LIST=12.1a
)

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "FAIL: $*"; exit 1; }
src_rev() { cat "$WORK/src/.gb10-rev" 2>/dev/null || echo norev; }
wheels_rev() { cat "$WORK/wheels/.gb10-rev" 2>/dev/null || echo norev; }
have_wheels() { compgen -G "$WORK/wheels/*.whl" >/dev/null; }
avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }

open_log() {
  LOGDIR="$WORK/logs/$(date +%Y%m%d-%H%M%S)-$1-$(src_rev)"
  mkdir -p "$LOGDIR" "$WORK/cache"
  exec > >(tee -a "$LOGDIR/run.log") 2>&1
  log "logs: $LOGDIR"
}

cmd_prep() {
  for img in "$BASE_IMAGE" "$BENCH_IMAGE"; do
    if docker image inspect "$img" >/dev/null 2>&1; then
      log "have $img"
    else
      log "pulling $img"
      docker pull "$img"
    fi
  done
  if [ -f "$MODEL_DIR/config.json" ]; then
    log "model already present: $MODEL_DIR"
  else
    log "downloading $MODEL_REPO to $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    # As the host user, so the checkpoint stays owned by whoever runs this.
    docker run --rm --memory 2g --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -e PIP_BREAK_SYSTEM_PACKAGES=1 -e HF_TOKEN -v "$MODEL_DIR:/out" "$BASE_IMAGE" bash -lc \
      "python3 -m pip install -q --user huggingface_hub && \
       python3 -c \"from huggingface_hub import snapshot_download as s; \
s('$MODEL_REPO', local_dir='/out')\""
  fi
  [ -f "$WORK/src/unieinfra/gb10/run.sh" ] || die "no source; run sync.sh first"
  log "prep done (source $(src_rev), wheels $(wheels_rev))"
}

# Make $IMAGE hold this source: install the prebuilt wheels into the base image
# in a memory-capped container, or, without wheels, build the Dockerfile here.
# Skipped when $IMAGE is already labelled with the same build.
ensure_image() {
  local want
  if have_wheels; then want="wheels-$(wheels_rev)"; else want="src-$(src_rev)"; fi
  if [ "$(docker image inspect -f '{{index .Config.Labels "gb10.build"}}' "$IMAGE" 2>/dev/null)" = "$want" ]; then
    log "$IMAGE already built from $want"
    return 0
  fi
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || die "no base image; run prep"

  if ! have_wheels; then
    [ "$(avail_gb)" -ge "$BUILD_MIN_FREE_GB" ] ||
      die "no wheels, and only $(avail_gb) GB free to build here (need $BUILD_MIN_FREE_GB); stop the service or use build_wheels.sh"
    log "building $IMAGE from source on this host (log: build.log)"
    docker build --label "gb10.build=$want" -f "$WORK/src/unieinfra/gb10/Dockerfile" \
      -t "$IMAGE" "$WORK/src" >"$LOGDIR/build.log" 2>&1 ||
      { tail -40 "$LOGDIR/build.log"; die "docker build"; }
    tail -2 "$LOGDIR/build.log"
    return 0
  fi

  docker rm -f "$INSTALL_CTR" >/dev/null 2>&1 || true
  log "installing wheels $(wheels_rev) into $IMAGE (log: install.log)"
  docker run --name "$INSTALL_CTR" --user root \
    --memory "${CHECK_MEM_GB}g" --memory-swap "${CHECK_MEM_GB}g" \
    -e PIP_BREAK_SYSTEM_PACKAGES=1 -v "$WORK/wheels:/wheels:ro" "$BASE_IMAGE" bash -lc '
      set -e
      apt-get update -qq
      apt-get install -y -qq --no-install-recommends libssl-dev libopenmpi-dev libnuma1 pkg-config
      python3 -m pip install /wheels/*.whl' >"$LOGDIR/install.log" 2>&1 ||
    { tail -30 "$LOGDIR/install.log"; docker rm -f "$INSTALL_CTR" >/dev/null; die "wheel install"; }
  docker commit \
    --change "LABEL gb10.build=$want" \
    --change 'ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas FLASHINFER_CUDA_ARCH_LIST=12.1a TORCH_CUDA_ARCH_LIST=12.1a' \
    --change 'CMD ["/bin/bash"]' \
    "$INSTALL_CTR" "$IMAGE" >/dev/null
  docker rm "$INSTALL_CTR" >/dev/null
  log "committed $IMAGE"
}

# Kill the test containers if host memory runs low. A cgroup cap bounds CPU
# allocations, but on unified memory GPU allocations may not be charged to it.
start_watchdog() {
  (
    while sleep 2; do
      avail_mb="$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)"
      if [ "$avail_mb" -lt "$GUARD_MB" ]; then
        log "WATCHDOG: MemAvailable ${avail_mb} MB < ${GUARD_MB} MB, killing test containers"
        docker kill "$CTR" "$INSTALL_CTR" >/dev/null 2>&1
      fi
    done
  ) &
  WATCHDOG_PID=$!
}

cmd_check() {
  [ -f "$WORK/src/unieinfra/gb10/level1.sh" ] || die "no source; run sync.sh first"
  if have_wheels && [ "$(src_rev)" != "$(wheels_rev)" ]; then
    log "WARNING: source $(src_rev) and wheels $(wheels_rev) come from different commits"
  fi
  open_log check

  log "host MemAvailable $(avail_gb) GB; container cap ${CHECK_MEM_GB} GB; watchdog floor ${GUARD_MB} MB"
  [ "$(avail_gb)" -ge $((CHECK_MEM_GB + 2)) ] ||
    die "need ${CHECK_MEM_GB} GB plus 2 GB headroom; lower GB10_CHECK_MEM_GB or free memory"

  start_watchdog
  trap 'kill "$WATCHDOG_PID" 2>/dev/null; docker rm -f "$CTR" "$INSTALL_CTR" >/dev/null 2>&1' EXIT

  ensure_image
  log "running level-1 checks (log: level1.log)"
  docker rm -f "$CTR" >/dev/null 2>&1 || true
  docker run --rm --name "$CTR" --gpus all --network none \
    --memory "${CHECK_MEM_GB}g" --memory-swap "${CHECK_MEM_GB}g" \
    "${RUNTIME_ENV[@]}" -e MAX_JOBS=1 \
    -v "$(readlink -f /usr/local/cuda):/hostcuda:ro" -e CUOBJDUMP=/hostcuda/bin/cuobjdump \
    -v "$WORK/src:/src:ro" -v "$WORK/cache:/root/.cache" \
    "$IMAGE" bash /src/unieinfra/gb10/level1.sh >"$LOGDIR/level1.log" 2>&1 ||
    log "level1.sh exited non-zero (killed by cap or watchdog?)"
  sed -n '/=== SUMMARY/,$p' "$LOGDIR/level1.log"
  log "full log: $LOGDIR/level1.log"
}

cmd_restore() {
  # Runs from the EXIT trap, possibly after the terminal is gone: start the
  # service before anything that writes output, and never stop on an error.
  set +e
  docker rm -f "$CTR" >/dev/null 2>&1
  docker start "$SERVICE" >/dev/null
  log "started $SERVICE; waiting for it to report healthy"
  for _ in $(seq 1 90); do
    state="$(docker inspect -f '{{.State.Health.Status}}' "$SERVICE" 2>/dev/null || echo none)"
    if [ "$state" = "healthy" ] || [ "$state" = "none" ]; then
      log "$SERVICE is $state"
      return 0
    fi
    sleep 10
  done
  log "WARNING: $SERVICE not healthy after 15 min; check: docker logs $SERVICE"
}

wait_ready() {
  local deadline=$((SECONDS + 3600))
  while ((SECONDS < deadline)); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then return 0; fi
    if ! docker exec "$CTR" bash -c '[ ! -f /logs/serve.pid ] || kill -0 "$(cat /logs/serve.pid)"' 2>/dev/null; then
      tail -50 "$LOGDIR/serve.log"
      die "server exited before becoming ready (log: $LOGDIR/serve.log)"
    fi
    sleep 15
  done
  die "server not ready after 60 min"
}

chat() {
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"$1\", \"temperature\": 0, \"max_tokens\": 256,
         \"chat_template_kwargs\": {\"enable_thinking\": false},
         \"messages\": [{\"role\": \"user\", \"content\": \"$2\"}]}" |
    python3 -c 'import sys, json; m = json.load(sys.stdin)["choices"][0]["message"]; print(m.get("content") or "[reasoning] " + str(m.get("reasoning_content")))'
}

# One `vllm bench serve` run against the test server. The client needs the GPU
# visible only because vLLM's CLI refuses to start without a device.
bench() {
  local name="$1" conc="$2" prefix="$3" inlen="$4" outlen="$5" n
  n=$((conc * 4 > BENCH_MIN_PROMPTS ? conc * 4 : BENCH_MIN_PROMPTS))
  docker run --rm --gpus all --network host --memory 4g --entrypoint vllm \
    -v "$MODEL_DIR:/models/model:ro" -v "$LOGDIR:/logs" "$BENCH_IMAGE" bench serve \
    --backend openai --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
    --model "$MODEL_ID" --tokenizer /models/model \
    --dataset-name random --random-prefix-len "$prefix" \
    --random-input-len "$inlen" --random-output-len "$outlen" --ignore-eos \
    --num-prompts "$n" --max-concurrency "$conc" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir /logs --result-filename "bench-$name-c$conc.json" \
    >"$LOGDIR/bench-$name-c$conc.txt" 2>&1 ||
    { log "WARNING: bench $name c=$conc failed (bench-$name-c$conc.txt)"; return 0; }
  python3 - "$LOGDIR/bench-$name-c$conc.json" "$name" "$conc" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  {sys.argv[2]:7s} c={int(sys.argv[3]):3d}  ok={r['completed']:4d}  "
      f"out={r['output_throughput']:8.1f} tok/s  "
      f"TTFT p50/p99={r['median_ttft_ms']:8.1f}/{r['p99_ttft_ms']:8.1f} ms  "
      f"TPOT p50/p99={r['median_tpot_ms']:6.1f}/{r['p99_tpot_ms']:6.1f} ms")
PY
}

cmd_window() {
  [ -f "$MODEL_DIR/config.json" ] || die "no model at $MODEL_DIR; run prep first"
  [ -f "$WORK/src/unieinfra/gb10/Dockerfile" ] || die "no source; run sync.sh first"
  docker image inspect "$BENCH_IMAGE" >/dev/null 2>&1 || die "no $BENCH_IMAGE; run prep first"
  open_log window
  [ -n "${TMUX:-}${STY:-}" ] || log "WARNING: not in tmux/screen; an SSH drop ends the window early"

  # Restart the service on exit only if this run is the one that stopped it;
  # a service someone stopped by hand stays stopped.
  if [ "$(docker inspect -f '{{.State.Running}}' "$SERVICE" 2>/dev/null)" = "true" ]; then
    trap cmd_restore EXIT
    trap 'exit 130' INT TERM HUP
    log "stopping $SERVICE (restarted when this run exits)"
    docker stop -t 60 "$SERVICE" >/dev/null
    sleep 5
  else
    trap 'docker rm -f "$CTR" >/dev/null 2>&1' EXIT
    trap 'exit 130' INT TERM HUP
    log "$SERVICE is not running; leaving it stopped"
  fi
  others="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
  [ -z "$others" ] || die "GPU still in use: $others"
  sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  free -g | head -2

  log "[1/4] image"
  ensure_image

  docker rm -f "$CTR" >/dev/null 2>&1 || true
  sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  docker run -d --name "$CTR" --gpus all --ipc=host --network host --shm-size 16g \
    "${RUNTIME_ENV[@]}" \
    -v "$MODEL_DIR:/models/model:ro" -v "$WORK/src:/src:ro" \
    -v "$WORK/cache:/root/.cache" -v "$LOGDIR:/logs" \
    "$IMAGE" sleep infinity >/dev/null

  log "[2/4] kernel plugins"
  docker exec "$CTR" python3 -m tokenspeed_kernel.plugins list | tee "$LOGDIR/plugins.txt"

  log "[3/4] serve and smoke test (first start JIT-compiles for sm_121; can take a while)"
  docker exec -d "$CTR" bash -lc "echo \$\$ >/logs/serve.pid; exec tokenspeed serve \
    /models/model --host 127.0.0.1 --port $PORT --gpu-memory-utilization $MEM_UTIL \
    $SERVE_ARGS >/logs/serve.log 2>&1"
  t0=$SECONDS
  wait_ready
  log "server ready in $((SECONDS - t0)) s"
  MODEL_ID="$(curl -sf "http://127.0.0.1:$PORT/v1/models" |
    python3 -c 'import sys, json; print(json.load(sys.stdin)["data"][0]["id"])')"
  a1="$(chat "$MODEL_ID" "What is the capital of France? Answer in one word.")"
  a2="$(chat "$MODEL_ID" "What is the capital of France? Answer in one word.")"
  printf 'answer: %s\n' "$a1" | tee "$LOGDIR/smoke.txt"
  grep -qi paris <<<"$a1" || die "smoke: expected Paris"
  # The second call can hit the prefix cache and take the extend path instead
  # of a full prefill, so a mismatch is a lead to chase, not proof of a bug.
  [ "$a1" = "$a2" ] || log "WARNING: temperature-0 answers differ: '$a2'"

  log "[4/4] benchmark: unique 1024-in prompts, then a shared 1024-token prefix + 256 new"
  for c in $CONCURRENCY; do bench unique "$c" 0 1024 128; done
  for c in $CONCURRENCY; do bench shared "$c" 1024 256 128; done

  log "done; results in $LOGDIR"
}

case "${1:-}" in
  prep) cmd_prep ;;
  check) cmd_check ;;
  window) cmd_window ;;
  restore) cmd_restore ;;
  *) echo "usage: $0 prep|check|window|restore" >&2; exit 2 ;;
esac
