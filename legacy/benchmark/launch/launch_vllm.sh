#!/usr/bin/env bash
# Start a vLLM OpenAI server for benchmarking on sm_120.
#
# vLLM keeps its OWN best defaults here (best-vs-best rule): attention is left
# on the prebuilt FLASH_ATTN backend vLLM auto-selects on this part. The only
# two overrides are sm_120 *correctness* shims, not tuning:
#   * VLLM_USE_FLASHINFER_SAMPLER=0 — FlashInfer's JIT arch check mis-detects
#     sm_120 ("FlashInfer requires sm75 or higher") and aborts startup.
#   * VLLM_SERVER_DEV_MODE=1 — exposes /reset_prefix_cache so the sweep can
#     flush between concurrency levels (harness hygiene, prefix cache stays ON).
# Set VLLM_ATTENTION_BACKEND explicitly to try another backend (e.g. TRITON_ATTN).
set -euo pipefail
MODEL="${MODEL:?set MODEL=/path/to/model}"
PORT="${PORT:-8001}"
GPU="${GPU:-1}"
VLLM_BIN="${VLLM_BIN:-/home/ubuntu/vllm-bench-venv/bin/vllm}"
# Optional: give vLLM the SAME quantization the UnieInfra arm is using, so a
# quantized comparison stays best-vs-best (e.g. QUANT=fp8 -> online W8A8 fp8
# from the same bf16 checkpoint). Empty = bf16.
QUANT="${QUANT:-}"
QARGS=""; [ -n "$QUANT" ] && QARGS="--quantization $QUANT"
GMU="${GMU:-0.85}"; MAXLEN="${MAXLEN:-4096}"
# Optional: load UnieInfra's out-of-tree admission policy through vLLM's own
# `--scheduler-cls` seam (see unieinfra-scheduler/). Empty = stock vLLM
# scheduler, i.e. this launcher is unchanged for every pre-existing arm.
# SCHED_OBJECTIVE selects the SLO objective the policy optimises for.
SCHED_CLS="${SCHED_CLS:-}"
SCHED_ARGS=""; [ -n "$SCHED_CLS" ] && SCHED_ARGS="--scheduler-cls $SCHED_CLS"
[ -n "${SCHED_OBJECTIVE:-}" ] && export UNIEINFRA_SCHED_OBJECTIVE="$SCHED_OBJECTIVE"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"

# --- record launch config (card, count, params) for reproducible result saving ---
NUM_GPUS=$(echo "$GPU" | tr ',' '\n' | grep -c .)
GPU_MODEL=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
RESDIR="$(cd "$(dirname "$0")/.." && pwd)/results"; mkdir -p "$RESDIR"
cat > "$RESDIR/vllm.launch.json" <<JSON
{"engine":"vllm","model":"$MODEL","served_model_name":"bench-model","port":$PORT,
 "gpu_ids":"$GPU","num_gpus":$NUM_GPUS,"gpu_model":"$GPU_MODEL","compute_cap":"sm_120",
 "dtype":"bfloat16","gpu_memory_utilization":$GMU,"max_model_len":$MAXLEN,
 "quantization":"$QUANT",
 "attention_backend":"$VLLM_ATTENTION_BACKEND","tp_size":$NUM_GPUS,
 "scheduler_cls":"$SCHED_CLS","sched_objective":"${SCHED_OBJECTIVE:-}",
 "sched_hold_steps":"${UNIEINFRA_SCHED_HOLD_STEPS:-}","sched_ttft_ms":"${UNIEINFRA_SCHED_TTFT_MS:-}",
 "sched_reserve_divisor":"${UNIEINFRA_SCHED_RESERVE_DIVISOR:-}","sched_cap":"${UNIEINFRA_SCHED_CAP:-}",
 "vllm_version":"$($VLLM_BIN --version 2>/dev/null | head -1)",
 "sm120_fix":"VLLM_USE_FLASHINFER_SAMPLER=0 to avoid flashinfer JIT sm_120 arch-check; attention uses prebuilt FLASH_ATTN (vLLM's own default on this part). VLLM_SERVER_DEV_MODE=1 to expose /reset_prefix_cache for per-level flush.",
 "flush_method":"/reset_prefix_cache before each concurrency level (prefix cache ON, seed=0)"}
JSON
echo "[launch] vllm on $NUM_GPUS x $GPU_MODEL (gpu ids=$GPU); config -> $RESDIR/vllm.launch.json"

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
  --served-model-name bench-model \
  --host 0.0.0.0 --port "$PORT" \
  --dtype bfloat16 --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
  $QARGS $SCHED_ARGS --trust-remote-code
