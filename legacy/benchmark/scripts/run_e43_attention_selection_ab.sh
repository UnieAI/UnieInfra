#!/usr/bin/env bash
# E43 — capability-based ATTENTION selection: does routing the backend choice
# through UnieInfra's kernel registry cost anything?
#
# Step A of docs/design/kernel-system-design-principles.md §5 replaces vLLM's
# hardcoded `_get_backend_priorities()` arch tree with a query against the
# capability registry. `dump_e43_attention_selection_table.py` already proves the
# *decision* is identical on sm_80/86/89/90/100/120 (150/150 shapes). This
# script answers the remaining question: does the machinery cost throughput?
#
# It should not. Selection runs once per distinct layer configuration inside
# `@cache`d `_cached_get_attn_backend`, at model construction — never on a hot
# path, never inside a captured CUDA graph. A measurable delta would mean
# something is wrong, not something is fast.
#
# ARMS (differ by exactly one env var; same fork, same build, same GEMM path)
#   attnstock  UNIEINFRA_ATTENTION=0 -> upstream's `CudaPlatform.get_attn_backend_cls`
#   attnunie   UNIEINFRA_ATTENTION=1 -> `resolve_attention_backend_cls` -> registry
#
# PROTOCOL (E34/E37 traps, re-applied)
#   * ONE PROCESS PER ARM. CUDA graphs are captured at engine construction, so
#     toggling the flag after startup replays the graph already captured.
#   * VLLM_DISABLE_COMPILE_CACHE=1 on both arms. vLLM's compile cache keys on
#     `VllmConfig`, not env vars.
#   * Arms ROTATED back-to-back so drift in box conditions is spread, not
#     confounded with one arm.
#   * The client runs INSIDE the container: it is bridge-networked with no
#     published ports, and the host venv lacks cbor2/msgspec/openai.
#   * Selection is DUMPED per PID on both arms. The stock arm's dump must read
#     `available: false, selection: []` — clean by evidence, not by assumption.
#
# Usage: GPU=2 run_e43_attention_selection_ab.sh [REPS]      (default 2)
set -uo pipefail

REPO=/workspace/roy/UnieInfra
CONTAINER=${CONTAINER:-unieinfra-vllm-build}
CONTAINER_IP=${CONTAINER_IP:-172.17.0.6}
GPU=${GPU:-auto}
# Never GPU 0: it belongs to another user.
# GPU 3 is left out: E41 is running there and its cleanup reaps by GPU uuid.
GPU_CANDIDATES=${GPU_CANDIDATES:-"1 2"}
PORT=${PORT:-8443}
REPS=${1:-2}
MODEL=${MODEL:-/workspace/models/Qwen/Qwen3-8B}
UNIE_BIN=/workspace/roy/venvs/unieinfra-vllm-venv/bin

RESULTS=${RESULTS:-$REPO/benchmark/results/e43}
SRVLOG=$REPO/benchmark/results/_srvlogs/e43
CONCS=${CONCS:-"1 32"}
OLEN=${OLEN:-256}
ILEN=${ILEN:-1024}
WARMUPS=${WARMUPS:-8}
# 0.50, not the usual 0.85. Four other agents share this box and one of them
# claimed 37 GiB on our GPU *between* the free-check and our KV allocation,
# OOM-ing the arm at warmup. Half the card still leaves ~28 GiB of KV, far more
# than C=32 x (1024+256) tokens needs, and it is applied IDENTICALLY to both
# arms so it cannot bias the comparison.
MEMUTIL=${MEMUTIL:-0.50}
ATTEMPTS=${ATTEMPTS:-3}

mkdir -p "$RESULTS" "$SRVLOG"
dex() { docker exec "$CONTAINER" bash -lc "$1"; }

PIDFILE=$SRVLOG/server.pid

# Cleanup discipline on a box with four other agents: NEVER kill a process this
# script did not start -- and make sure THEIR cleanup cannot kill ours either.
# The server is launched as `python -m vllm.entrypoints.cli.main serve` rather
# than via the `unieinfra`/`vllm` console script, because the concurrently
# running E41 harness does a container-wide `pkill -f '[u]nieinfra serve'` and a
# host-side `pkill -f "[v]llm serve $MODEL"`; the container shares the host PID
# namespace, so both patterns reach our process. Neither matches this cmdline. The E41-era reaper decided ownership from /proc
# (venv path / container id) and would have killed another agent's
# `bench_quant_arm.py` EngineCore that grabbed this GPU one second after our
# free-check -- it only survived because it runs as root and we do not. Instead
# the server is started under `setsid`, so its pid is a process-GROUP leader and
# `kill -- -PID` takes the server plus its EngineCore children and nothing else.
kill_server() {
  local pid
  pid=$(dex "cat $PIDFILE 2>/dev/null" | tr -d '\r\n')
  [ -n "$pid" ] || return 0
  dex "kill -- -$pid 2>/dev/null; true"
  for _ in $(seq 1 60); do
    dex "kill -0 -- -$pid 2>/dev/null" || break
    sleep 2
  done
  dex "kill -9 -- -$pid 2>/dev/null; rm -f $PIDFILE; true"
  sleep 5
}

# Refuse to measure next to somebody else's job. `min_free` is checked twice,
# five seconds apart, because GPUs on this box get claimed between the check and
# the launch (that is exactly what happened on the first attempt).
wait_gpu_free() {
  local used used2
  for i in $(seq 1 180); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    if [ "$used" -lt 2000 ]; then
      sleep 5
      used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
      [ "$used2" -lt 2000 ] && { [ "$i" -gt 1 ] && echo "  [waited $((i*10))s for GPU $GPU]"; return 0; }
    fi
    [ "$i" = 1 ] && echo "  [CONTENTION] GPU $GPU holds ${used} MiB — waiting up to 30 min"
    sleep 10
  done
  echo "  [CONTENTION UNRESOLVED] GPU $GPU still busy; results from here are suspect"
  return 1
}

# start_server <arm> <log>
start_server() {
  local arm=$1 log=$2 attn=1
  [ "$arm" = "attnstock" ] && attn=0
  : > "$log"
  docker exec -d "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1
    export UNIEINFRA_ATTENTION=$attn
    export UNIEINFRA_ATTENTION_LOG_SELECTION=1
    export UNIEINFRA_ATTENTION_SELECTION_DUMP=$SRVLOG/sel_${arm}
    setsid bash -c 'echo \$\$ > $PIDFILE; exec $UNIE_BIN/python3 -m vllm.entrypoints.cli.main serve $MODEL \
      --host 0.0.0.0 --port $PORT --served-model-name bench-model \
      --max-model-len 4096 --gpu-memory-utilization $MEMUTIL' >> $log 2>&1"
  for _ in $(seq 1 400); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  [ready] $arm"; return 0; }
    grep -qE "Traceback \(most recent|CUDA out of memory|Engine core initialization failed|Killed" "$log" 2>/dev/null && {
      echo "  [FAIL] $arm — see $log"; tail -25 "$log"; return 1; }
    sleep 4
  done
  echo "  [TIMEOUT] $arm"; return 1
}

# greedy_probe <arm> — coherence check, not a metric. Greedy, so it is
# reproducible; E37 used the same prompt so the answers are comparable.
greedy_probe() {
  local arm=$1
  dex "curl -s http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"bench-model\",\"prompt\":\"The capital of Italy is\",\"max_tokens\":48,\"temperature\":0}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"choices\"][0][\"text\"])'" \
    | tee "$RESULTS/probe_${arm}.txt"
}

run_one() {  # run_one <arm> <rep> <C>
  local arm=$1 rep=$2 c=$3 np
  np=$(( c * 8 )); [ "$np" -gt 512 ] && np=512
  dex "
    cd $REPO
    $UNIE_BIN/vllm bench serve --backend openai \
      --host 127.0.0.1 --port $PORT --model bench-model --tokenizer $MODEL \
      --dataset-name random --random-input-len $ILEN --random-output-len $OLEN \
      --ignore-eos --num-prompts $np --max-concurrency $c --num-warmups $WARMUPS \
      --temperature 0 --seed 0 --disable-tqdm \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99 \
      --save-result --result-dir $RESULTS/e43_${arm}_r${rep} \
      --result-filename c${c}.json --label e43_${arm}_r${rep}
  " >> "$SRVLOG/bench_${arm}_r${rep}_c${c}.log" 2>&1
}

trap 'echo "[trap] cleaning up"; kill_server' EXIT INT TERM

# Pick a GPU ONCE and keep it for every arm: an A/B split across two parts is
# not an A/B. Four other agents are working on this box, so "free" is checked
# twice, five seconds apart, before a candidate is claimed.
pick_gpu() {
  local g used used2
  for i in $(seq 1 360); do
    for g in $GPU_CANDIDATES; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      [ "$used" -lt 2000 ] || continue
      sleep 5
      used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      [ "$used2" -lt 2000 ] && { GPU=$g; echo "  [claimed GPU $g after $((i*10))s]"; return 0; }
    done
    [ "$i" = 1 ] && echo "  [CONTENTION] no free GPU among {$GPU_CANDIDATES} — waiting up to 60 min"
    sleep 10
  done
  echo "  [CONTENTION UNRESOLVED] no GPU came free"; return 1
}

echo "=== E43 attention-selection A/B ($(date -u +%FT%TZ)) ==="
[ "$GPU" = "auto" ] && { pick_gpu || exit 1; }
echo "  GPU=$GPU"
kill_server
wait_gpu_free || exit 1

for rep in $(seq 1 "$REPS"); do
  # Rotate arm order between reps: stock,unie then unie,stock.
  if [ $((rep % 2)) -eq 1 ]; then order="attnstock attnunie"; else order="attnunie attnstock"; fi
  for arm in $order; do
    echo "--- rep $rep arm $arm $(date -u +%FT%TZ)"
    kill_server
    wait_gpu_free || exit 1
    ok=0
    for attempt in $(seq 1 "$ATTEMPTS"); do
      start_server "$arm" "$SRVLOG/${arm}_r${rep}.log" && { ok=1; break; }
      echo "  [retry $attempt/$ATTEMPTS] $arm — a co-tenant took the GPU mid-startup"
      kill_server
      wait_gpu_free || exit 1
    done
    [ "$ok" = 1 ] || { echo "  [GIVING UP] $arm after $ATTEMPTS attempts"; exit 1; }
    [ "$rep" = 1 ] && greedy_probe "$arm"
    for c in $CONCS; do run_one "$arm" "$rep" "$c"; done
    kill_server
  done
done

echo
echo "=== summary ==="
python3 - "$RESULTS" <<'PY'
import glob, json, os, statistics, sys
root = sys.argv[1]
rows = {}
for path in glob.glob(os.path.join(root, "e43_*_r*", "c*.json")):
    arm = os.path.basename(os.path.dirname(path)).split("_")[1]
    c = int(os.path.basename(path)[1:-5])
    d = json.load(open(path))
    rows.setdefault((c, arm), []).append(d["output_throughput"])
concs = sorted({c for c, _ in rows})
print(f"{'C':>4} {'attnstock':>22} {'attnunie':>22} {'delta':>9}")
for c in concs:
    a = rows.get((c, "attnstock"), [])
    b = rows.get((c, "attnunie"), [])
    if not a or not b:
        continue
    ma, mb = statistics.median(a), statistics.median(b)
    print(f"{c:>4} {ma:>10.2f} [{min(a):.2f}-{max(a):.2f}] "
          f"{mb:>10.2f} [{min(b):.2f}-{max(b):.2f}] {100*(mb-ma)/ma:>+8.2f}%")
PY

echo
echo "=== selection evidence ==="
for f in "$SRVLOG"/sel_*.attention.*.json; do
  [ -e "$f" ] || continue
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{sys.argv[1].split('/')[-1]}: available={d['available']} "
      f"counters={d['counters']} decisions={len(d['selection'])}")
for e in d["selection"]:
    print(f"   {e['mode']} hs={e['head_size']} {e['dtype']} kv={e['kv_cache_dtype']} "
          f"block={e['block_size']} swa={e['has_sliding_window']} -> {e['backend']} ({e['kernel']})")
PY
done
