#!/usr/bin/env bash
# E47 — THE MISSING TABLE: stock vLLM vs UnieInfra with everything of ours ON.
#
# Every recent experiment measured a feature vLLM already had (its own MTP, its
# own prefix cache, its own grammar). None of them answered the only question
# that matters for the integration: *what is UnieInfra worth over stock vLLM?*
# This script runs that comparison as ONE sweep, three precision arms.
#
# ARMS (per precision) --------------------------------------------------------
#   stock     host venv /home/ubuntu/vllm-bench-venv, vLLM 0.25.1 from PyPI.
#   forkbase  the same fork with every UNIEINFRA_* seam switched OFF, so the
#             engine runs upstream's own paths. This arm exists to SPLIT the
#             delta: (stock -> forkbase) is upstream drift between the two
#             snapshots, (forkbase -> unie) is OUR work. Without it, a
#             stock-vs-ours number cannot be attributed to anything.
#   unie      the same fork with everything of ours enabled: UNIEINFRA_GEMM
#             (dense decode GEMV, E37), UNIEINFRA_QUANT (the fp8 / nvfp4 decode
#             kernels through the engine's OWN quantized dispatch -- the direct
#             seam that replaced E38's `--quantization unieinfra` plugin) and
#             UNIEINFRA_ATTENTION (capability-based backend selection).
#
# WHY A FROZEN TREE ------------------------------------------------------------
# The fork is installed EDITABLE, so the installed engine is the repo tree, and
# another agent is actively editing that tree (the direct quant seam and the
# attention selector both landed during this experiment's pilot). A four-hour
# A/B against a moving definition of "arm B" is not an A/B. So both fork arms
# run against a byte-frozen copy of `unieinfra-vllm/vllm` + `unieinfra_kernel`
# taken once, put first on PYTHONPATH. setuptools' editable finder is APPENDED
# to sys.meta_path, so a PYTHONPATH copy wins over it -- verified, not assumed.
#
# WHAT IS DELIBERATELY *NOT* VARIED -------------------------------------------
#   * MTP / speculative decoding: OFF everywhere. It is vLLM's feature and it is
#     present in both arms; mixing it in is exactly what obscured the picture.
#   * Prefix caching: left at each engine's default (ON), never flushed, never
#     configured differently between arms.
#   * Sampler: stock vLLM 0.25.1 CANNOT START on sm_120 without
#     VLLM_USE_FLASHINFER_SAMPLER=0 (verified again here: "FlashInfer requires
#     GPUs with sm75 or higher"). That is stock's own best *working* default,
#     so it is applied to the stock arm only, and the forkbase arm is what
#     accounts for it.
#
# CLIENT ----------------------------------------------------------------------
#   ONE client for both arms: the stock `vllm bench serve` on the HOST. The
#   container is bridge-networked with no published ports, but host->container
#   is routable on docker0 (172.17.0.x), so the same client process can drive
#   both servers. Using one client keeps metric computation identical.
#
# PROTOCOL (E35 discipline) ---------------------------------------------------
#   N>=3 repeats, arms ROTATED back-to-back within each round, median with
#   [min-max] spread, and a delta inside the spread is reported as a TIE.
#
# Raw output: benchmark/results/e47/e47_<precision>_<arm>_r<rep>/c<C>.json
set -uo pipefail

REPO=/workspace/roy/UnieInfra
CONTAINER=${CONTAINER:-unieinfra-vllm-build}
CONTAINER_IP=${CONTAINER_IP:-172.17.0.6}
GPU=${GPU:-3}
PORT_STOCK=${PORT_STOCK:-8401}
PORT_UNIE=${PORT_UNIE:-8402}

FROZEN=${FROZEN:-/workspace/roy/e47_frozen}
STOCK_BIN=/home/ubuntu/vllm-bench-venv/bin
UNIE_BIN=/workspace/roy/venvs/unieinfra-vllm-venv/bin
CLIENT=$STOCK_BIN/vllm          # the ONE client, host side, stock 0.25.1

MODEL_BF16=${MODEL_BF16:-/workspace/models/Qwen/Qwen3-8B}
MODEL_NVFP4=${MODEL_NVFP4:-/workspace/models/nvidia/Qwen3-8B-NVFP4}

RESULTS=${RESULTS:-$REPO/benchmark/results/e47}
SRVLOG=$REPO/benchmark/results/_srvlogs/e47

OLEN=${OLEN:-256}
ILEN=${ILEN:-1024}
CONCS=${CONCS:-"1 16 32 64 128 256"}
REPS=${REPS:-3}
WARMUPS=${WARMUPS:-8}
PRECISIONS=${PRECISIONS:-"bf16 fp8 nvfp4"}
# arms per precision; bf16 gets the three-way split, quant arms two-way
ARMS_BF16=${ARMS_BF16:-"stock forkbase unie"}
ARMS_QUANT=${ARMS_QUANT:-"stock forkbase unie"}
# 8 prompts per concurrent slot, capped: at C=256 the uncapped 2048x(1024+256)
# tokens costs ~3.5 min per cell and buys no extra stability.
MAX_PROMPTS=${MAX_PROMPTS:-512}

mkdir -p "$RESULTS" "$SRVLOG"

dex() { docker exec "$CONTAINER" bash -lc "$1"; }

GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
CTR_ID=$(dex 'cat /etc/hostname' | tr -d '\r\n')

# PIDs that were already on our GPU before we ever launched a server. They are
# SOMEBODY ELSE'S and are never touched. This list is the whole safety mechanism.
#
# The first version of this function decided ownership by matching /proc against
# our venv paths and our container id -- and killed another agent's
# `experiment/e45_scratch/sweep.py`, which happened to run inside the same
# container and therefore looked like ours. Sharing a container is not owning a
# process. Only "this PID did not exist before I started" is.
FOREIGN_PIDS=""
snapshot_foreign() {
  FOREIGN_PIDS=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
    | awk -F', ' -v u="$GPU_UUID" '$2==u{print $1}' | tr '\n' ' ')
  [ -n "$FOREIGN_PIDS" ] && echo "  [foreign on GPU $GPU, will not be touched] $FOREIGN_PIDS"
}

# vLLM's `VLLM::EngineCore` worker survives a pkill of its `vllm serve` parent
# and keeps the whole KV pool allocated -- that orphan is what made the first
# E47 pilot fail with "Free memory on device cuda:0 (11.97/94.97 GiB)". Reap
# only PIDs that appeared on our GPU after our snapshot.
kill_gpu_orphans() {
  local pid uuid cmd
  while read -r pid uuid; do
    [ "$uuid" = "$GPU_UUID" ] || continue
    case " $FOREIGN_PIDS " in *" $pid "*) continue ;; esac
    # Second, independent condition: it must also LOOK like one of our servers.
    # Both tests must pass, so a foreign job that starts mid-run is still safe.
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$cmd" in
      *VLLM::EngineCore*|*"vllm serve"*|*"unieinfra serve"*) ;;
      *) continue ;;
    esac
    echo "  [reap] killing our orphan pid=$pid on GPU $GPU"
    kill -9 "$pid" 2>/dev/null
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',')
}

kill_servers() {
  # [v] bracket trick: a plain pattern would match this very command line.
  pkill -f "[v]llm serve $MODEL_BF16"   >/dev/null 2>&1
  pkill -f "[v]llm serve $MODEL_NVFP4"  >/dev/null 2>&1
  dex "pkill -f '[u]nieinfra serve' >/dev/null 2>&1; true"
  for _ in $(seq 1 60); do
    local host_alive=0 ctr_alive=0
    pgrep -f "[v]llm serve /workspace/models" >/dev/null 2>&1 && host_alive=1
    dex "pgrep -f '[u]nieinfra serve' >/dev/null 2>&1" && ctr_alive=1
    [ $host_alive -eq 0 ] && [ $ctr_alive -eq 0 ] && break
    sleep 2
  done
  sleep 5
  kill_gpu_orphans
  sleep 5
}

# Refuse to start a measurement next to somebody else's job: an arm that shares
# the part is not a measurement. Waits, then reports honestly if it gives up.
wait_gpu_free() {
  local used
  for i in $(seq 1 180); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$used" -lt 2000 ] && { [ "$i" -gt 1 ] && echo "  [waited $((i*10))s for GPU $GPU to clear]"; return 0; }
    [ "$i" = 1 ] && echo "  [CONTENTION] GPU $GPU holds ${used} MiB — waiting up to 30 min"
    sleep 10
  done
  echo "  [CONTENTION UNRESOLVED] GPU $GPU still busy; results from here are suspect"
  return 1
}

# model_for <precision>
model_for() { [ "$1" = "nvfp4" ] && echo "$MODEL_NVFP4" || echo "$MODEL_BF16"; }

# sm_120 shims for the NVFP4 checkpoint, applied IDENTICALLY to both arms:
# vLLM's `auto` NVFP4 backend picks FlashInfer whose JIT rejects major==12, and
# the checkpoint's fp8 KV cache makes it auto-select FlashInfer attention too.
nvfp4_backends() { echo "--linear-backend cutlass --attention-backend TRITON_ATTN"; }

# start_server <precision> <arm> <logfile> -> 0 ready / 1 failed
start_server() {
  local prec=$1 arm=$2 log=$3
  local model quant_args="" extra=""
  model=$(model_for "$prec")
  : > "$log"

  case "$prec" in
    fp8)   quant_args="--quantization fp8" ;;
    nvfp4) quant_args="--quantization modelopt_fp4 $(nvfp4_backends)" ;;
  esac

  if [ "$arm" = "stock" ]; then
    CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1 \
      VLLM_USE_FLASHINFER_SAMPLER=0 \
      nohup $STOCK_BIN/vllm serve "$model" --host 0.0.0.0 --port $PORT_STOCK \
        --served-model-name bench-model --max-model-len 4096 \
        --gpu-memory-utilization 0.85 $quant_args >> "$log" 2>&1 &
  else
    # Both fork arms take the SAME --quantization flag (the engine's own name);
    # they differ only in whether our seams are switched on. That is what makes
    # the two arms hold bit-identical weights and differ by exactly our kernels.
    # gemm/quant/attn switches per arm. `kernels` and `attn` are diagnostic arms
    # that split "everything of ours" into its two independent halves.
    local g q a
    case "$arm" in
      forkbase) g=0; q=0; a=0 ;;
      kernels)  g=1; q=1; a=0 ;;   # decode GEMV + quantized decode kernels only
      attn)     g=0; q=0; a=1 ;;   # attention backend selection only
      *)        g=1; q=1; a=1 ;;   # unie: everything of ours
    esac
    docker exec -d "$CONTAINER" bash -lc "
      export PYTHONPATH=$FROZEN
      export CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1
      export UNIEINFRA_GEMM=$g UNIEINFRA_QUANT=$q UNIEINFRA_ATTENTION=$a
      export UNIEINFRA_KERNEL_SELECTION_DUMP=$SRVLOG/sel_${prec}_${arm}
      export UNIEINFRA_ATTENTION_SELECTION_DUMP=$SRVLOG/attn_${prec}_${arm}
      exec $UNIE_BIN/unieinfra serve $model --host 0.0.0.0 --port $PORT_UNIE \
        --served-model-name bench-model --max-model-len 4096 \
        --gpu-memory-utilization 0.85 $quant_args >> $log 2>&1"
  fi

  # NOTE: the fork logs a *warning* whose body contains "Traceback (most recent
  # call last)" -- `vllm/v1/attention/unieinfra_selection.py` cannot import
  # `SelectionObjective` from the installed `unieinfra_kernel` and degrades to
  # upstream's selector. Matching that as a failure aborted a healthy server in
  # the E47 pilot, so the failure patterns exclude it.
  for _ in $(seq 1 400); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  [ready] $prec/$arm"; return 0; }
    grep -vE "unieinfra_selection|WARNING" "$log" 2>/dev/null \
      | grep -qE "Traceback \(most recent|CUDA out of memory|Engine core initialization failed|Killed" && {
      echo "  [FAIL] $prec/$arm — see $log"; grep -E "Error|error:|ValueError|RuntimeError" "$log" | tail -8; return 1; }
    sleep 4
  done
  echo "  [TIMEOUT] $prec/$arm"; return 1
}

# run_one <precision> <arm> <rep> <C>
run_one() {
  local prec=$1 arm=$2 rep=$3 c=$4
  local host port model np label outdir
  if [ "$arm" = "stock" ]; then host=127.0.0.1; port=$PORT_STOCK
  else host=$CONTAINER_IP; port=$PORT_UNIE; fi
  model=$(model_for "$prec")
  np=$(( c * 8 )); [ "$np" -lt 8 ] && np=8; [ "$np" -gt "$MAX_PROMPTS" ] && np=$MAX_PROMPTS
  label="e41_${prec}_${arm}_r${rep}"
  outdir="$RESULTS/$label"
  mkdir -p "$outdir"
  echo "  -> $prec/$arm rep=$rep C=$c np=$np"
  # Box audit per cell. Another agent is cycling its own servers on this GPU, so
  # a cell that ran with more than one process resident is a cell to distrust --
  # recorded rather than assumed away.
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader \
    | grep "$GPU_UUID" > "$outdir/gpu_c${c}.before" 2>/dev/null
  HF_HUB_DISABLE_PROGRESS_BARS=1 $CLIENT bench serve \
    --backend openai --host "$host" --port "$port" --model bench-model \
    --tokenizer "$model" \
    --dataset-name random --random-input-len $ILEN --random-output-len $OLEN \
    --ignore-eos \
    --num-prompts $np --max-concurrency $c --num-warmups $WARMUPS \
    --temperature 0 --seed 0 --disable-tqdm \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99 \
    --save-result --result-dir "$outdir" --result-filename "c${c}.json" \
    --label "$label" > "$SRVLOG/bench_${label}_c${c}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "     [bench FAILED rc=$rc] $SRVLOG/bench_${label}_c${c}.log"; tail -12 "$SRVLOG/bench_${label}_c${c}.log"
  else
    grep -E "Output token throughput|Mean TTFT|P99 TTFT|Mean TPOT|P99 TPOT" \
      "$SRVLOG/bench_${label}_c${c}.log" | sed 's/^/       /'
  fi
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader \
    | grep "$GPU_UUID" > "$outdir/gpu_c${c}.after" 2>/dev/null
  # Per-cell snapshot of the kernel selection evidence (shape table + the
  # fast/delegated counters). Cumulative within a server, so the aggregator
  # reads the DELTA between consecutive cells.
  if [ "$arm" != "stock" ]; then
    dex "for f in $SRVLOG/sel_${prec}_${arm}.*.json; do [ -e \"\$f\" ] || continue; cp \"\$f\" \"$outdir/sel_c${c}.\$(basename \$f)\"; done; true" >/dev/null 2>&1
  fi
}

echo "=== E47 stock-vs-UnieInfra — GPU $GPU, precisions: $PRECISIONS, N=$REPS ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  gpu /'
snapshot_foreign

for prec in $PRECISIONS; do
  [ "$prec" = "bf16" ] && ALL_ARMS="$ARMS_BF16" || ALL_ARMS="$ARMS_QUANT"
  for rep in $(seq 1 "$REPS"); do
    # rotate arm order every round so slow drift cannot systematically favour one
    if [ $(( rep % 2 )) -eq 1 ]; then ARMS="$ALL_ARMS"; else ARMS=$(echo "$ALL_ARMS" | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for arm in $ARMS; do
      echo "--- $prec round $rep / arm $arm  $(date -u +%FT%TZ) ---"
      # Another agent is cycling its own servers on this GPU, so "free" can stop
      # being true between the check and the allocation. Retry rather than drop
      # the arm: a missing arm is a hole in the table, and holes get filled with
      # guesses.
      ok=0
      for attempt in 1 2 3 4 5; do
        kill_servers
        wait_gpu_free
        if start_server "$prec" "$arm" "$SRVLOG/${prec}_${arm}_r${rep}.log"; then ok=1; break; fi
        echo "  [retry $attempt] $prec/$arm r$rep did not start"
      done
      [ $ok -eq 1 ] || { echo "skip $prec/$arm r$rep after 5 attempts"; continue; }
      for c in $CONCS; do run_one "$prec" "$arm" "$rep" "$c"; done
      kill_servers
    done
  done
done

echo "=== E47 done $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
