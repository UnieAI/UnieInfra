#!/usr/bin/env bash
# Launch SGLang 0.5.16 on the HOST (its own venv) on GPU 3, sm_120 fix applied.
# SGLang's OWN best defaults; only attention-backend triton (sm_120 constraint,
# same as all engines here). No hand-tuning of scheduler/batch caps.
set -uo pipefail
PORT="${PORT:-8002}"; GPU="${GPU:-3}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen/Qwen3-8B}"
# Optional: same quantization as the UnieInfra arm under test, so a quantized
# comparison stays best-vs-best (e.g. QUANT=fp8). Empty = bf16.
QUANT="${QUANT:-}"
QARGS=""; [ -n "$QUANT" ] && QARGS="--quantization $QUANT"
GMU="${GMU:-0.85}"; MAXLEN="${MAXLEN:-4096}"   # mem-fraction-static / context-length
VENV=/home/ubuntu/sglang-venv
CU13="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="$CU13"
export PATH="$CU13/bin:$PATH"
export LD_LIBRARY_PATH="$CU13/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
LOG="${LOG:-/workspace/roy/UnieInfra/benchmark/results/_srvlogs/sglang.log}"

# --- record launch config (card, count, params) for reproducible result saving ---
GPU_MODEL=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
RESDIR="/workspace/roy/UnieInfra/benchmark/results"; mkdir -p "$RESDIR"
cat > "$RESDIR/sglang.launch.json" <<JSON
{"engine":"sglang","model":"$MODEL_PATH","served_model_name":"bench-model","port":$PORT,"quantization":"$QUANT",
 "gpu_ids":"$GPU","num_gpus":1,"gpu_model":"$GPU_MODEL","compute_cap":"sm_120",
 "dtype":"bfloat16","mem_fraction_static":0.85,"context_length":4096,
 "attention_backend":"triton","tp_size":1,
 "sglang_version":"$("$VENV/bin/python" -c 'import sglang;print(sglang.__version__)' 2>/dev/null)",
 "sm120_fix":"System nvcc is CUDA 12.0 (rejects compute_120f). Set CUDA_HOME + PATH to the venv-bundled CUDA 13.x toolkit (nvidia/cu13) so sgl-kernel JIT compiles sm_120f. Launched --attention-backend triton to avoid flashinfer attention JIT on sm_120.",
 "flush_method":"/flush_cache before each concurrency level (RadixCache prefix cache ON, seed=0)"}
JSON
echo "[launch] sglang on 1 x $GPU_MODEL (gpu ids=$GPU); config -> $RESDIR/sglang.launch.json"

exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name bench-model \
  --host 0.0.0.0 --port "$PORT" \
  --dtype bfloat16 --mem-fraction-static 0.85 --context-length 4096 \
  $QARGS --attention-backend triton --trust-remote-code --random-seed 0 \
  > "$LOG" 2>&1
