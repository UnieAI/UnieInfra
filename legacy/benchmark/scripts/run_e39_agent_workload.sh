#!/usr/bin/env bash
# E39 — AGENT-WORKLOAD benchmark: does MTP pay off on agent-shaped work?
#
# Answers: "published results claim large gains from MTP, but our benchmarks
# (RandomDataset ILEN 1024 / OLEN 256) never show them. Is the gain not real,
# or have we only measured the one scenario where it cannot appear?"
#
# Design (E35 discipline):
#   * two arms, SAME launch params except --speculative-config:
#       base = no speculative decoding
#       mtp  = in-checkpoint MTP head, num_speculative_tokens=3 (k=3, 4 positions)
#   * three datasets: spec_bench + gsm8k (agent-shaped) and random (the CONTROL,
#     which reproduces our entire E1-E38 history: ILEN 1024 / OLEN 256).
#   * concurrency 1 / 16 / 64, N=3 repeats, arms ROTATED back-to-back between
#     rounds so slow drift cannot favour one arm.
#   * greedy (temperature 0) on both arms -> MTP is output-lossless, so the two
#     arms emit the same tokens and throughput is directly comparable.
#
# Everything runs INSIDE the unieinfra-vllm-build container: that container has
# torch 2.11.0+cu129 matching the compiled extensions, and it is on a bridge
# network with no published ports, so the bench client must run there too.
#
# Raw output: benchmark/results/unieinfra/e39_<arm>_<dataset>_r<rep>/c<C>.json
set -uo pipefail

CONTAINER=${CONTAINER:-unieinfra-vllm-build}
GPU=${GPU:-2}
PORT=${PORT:-8311}
MODEL=${MODEL:-/workspace/models/Qwen/Qwen3.5-9B}
REPO=/workspace/roy/UnieInfra
VENV=/workspace/roy/venvs/unieinfra-vllm-venv/bin
RESULTS=$REPO/benchmark/results/unieinfra
SRVLOG=$REPO/benchmark/results/_srvlogs/e39
SPECBENCH=/workspace/roy/bench-data/specbench_question.jsonl

OLEN=${OLEN:-256}
ILEN=${ILEN:-1024}          # random control only — matches E1-E38 history
CONCS=${CONCS:-"1 16 64"}
REPS=${REPS:-3}
WARMUPS=${WARMUPS:-8}
SPEC_K=${SPEC_K:-3}
DATASETS=${DATASETS:-"specbench gsm8k random"}

mkdir -p "$SRVLOG"

dex() { docker exec "$CONTAINER" bash -lc "$1"; }

kill_server() {
  # NOTE the [u] bracket trick: `pgrep -f 'unieinfra serve'` would otherwise match
  # the `docker exec ... bash -lc "pgrep -f 'unieinfra serve'"` wrapper's OWN command
  # line and never report the server as gone.
  dex "pkill -f '[u]nieinfra serve' >/dev/null 2>&1; true"
  for _ in $(seq 1 60); do
    dex "pgrep -f '[u]nieinfra serve' >/dev/null 2>&1" || break
    sleep 2
  done
  sleep 5
}

# start_server <arm>
start_server() {
  local arm=$1 spec="" log="$SRVLOG/${arm}.log"
  if [ "$arm" = "mtp" ]; then
    spec="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K}'"
  fi
  : > "$log"
  docker exec -d "$CONTAINER" bash -lc "
    export CUDA_VISIBLE_DEVICES=$GPU VLLM_DISABLE_COMPILE_CACHE=1
    exec $VENV/unieinfra serve $MODEL --port $PORT \
      --max-model-len 4096 --gpu-memory-utilization 0.85 --served-model-name q35 \
      $spec > $log 2>&1"
  for _ in $(seq 1 300); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  [ready] arm=$arm"; return 0; }
    grep -qE "Traceback \(most recent|CUDA out of memory|Killed" "$log" 2>/dev/null && {
      echo "  [FAIL] arm=$arm — see $log"; tail -20 "$log"; return 1; }
    sleep 4
  done
  echo "  [TIMEOUT] arm=$arm"; return 1
}

# run_one <arm> <dataset> <rep> <concurrency>
run_one() {
  local arm=$1 ds=$2 rep=$3 c=$4
  local label="e39_${arm}_${ds}_r${rep}"
  local outdir="$RESULTS/$label"
  local np=$(( c * 8 )); [ "$np" -lt 8 ] && np=8
  local dsargs

  case "$ds" in
    specbench) dsargs="--dataset-name spec_bench --dataset-path $SPECBENCH --spec-bench-output-len $OLEN" ;;
    gsm8k)     dsargs="--dataset-name hf --dataset-path openai/gsm8k --hf-subset main --hf-split test --hf-output-len $OLEN" ;;
    # The CONTROL: identical to every benchmark E1-E38 ever ran.
    random)    dsargs="--dataset-name random --random-input-len $ILEN --random-output-len $OLEN --ignore-eos" ;;
    *) echo "unknown dataset $ds"; return 1 ;;
  esac

  mkdir -p "$outdir"
  echo "  -> arm=$arm ds=$ds rep=$rep C=$c np=$np"
  dex "cd $REPO && HF_HUB_DISABLE_PROGRESS_BARS=1 $VENV/unieinfra bench serve \
      --backend openai --host 127.0.0.1 --port $PORT --model q35 \
      --tokenizer $MODEL \
      $dsargs \
      --num-prompts $np --max-concurrency $c --num-warmups $WARMUPS \
      --temperature 0 --seed 0 --disable-tqdm \
      --save-result --result-dir $outdir --result-filename c${c}.json \
      --label $label" > "$SRVLOG/bench_${label}_c${c}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "     [bench FAILED rc=$rc] see $SRVLOG/bench_${label}_c${c}.log"
    tail -15 "$SRVLOG/bench_${label}_c${c}.log"
  else
    grep -E "Output token throughput|Acceptance length|Mean TTFT|Mean TPOT" \
      "$SRVLOG/bench_${label}_c${c}.log" | sed 's/^/       /'
  fi
}

echo "=== E39 agent-workload benchmark — GPU $GPU, model $MODEL, k=$SPEC_K ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'

for rep in $(seq 1 "$REPS"); do
  # rotate arm order every round so drift cannot systematically favour one arm
  if [ $(( rep % 2 )) -eq 1 ]; then ARMS="base mtp"; else ARMS="mtp base"; fi
  for arm in $ARMS; do
    echo "--- round $rep / arm $arm ---"
    kill_server
    start_server "$arm" || { echo "skip arm=$arm rep=$rep"; continue; }
    for ds in $DATASETS; do
      for c in $CONCS; do
        run_one "$arm" "$ds" "$rep" "$c"
      done
    done
    kill_server
  done
done

echo "=== done; summarizing ==="
for d in "$RESULTS"/e39_*/; do
  [ -d "$d" ] || continue
  python3 "$REPO/benchmark/summarize.py" "$d" "$d/summary.csv" unieinfra >/dev/null 2>&1
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
