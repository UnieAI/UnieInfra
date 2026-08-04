"""E50: probe which KV cache dtypes actually work on sm_120 with TRITON_ATTN.

Runs one dtype per process invocation so a crash isolates cleanly.
Prints, in machine-greppable form (line prefixed ``E50RESULT``):
  - the attention backend actually selected (not the one requested)
  - the real KV cache tensor dtype / shape / bytes-per-token
  - num_gpu_blocks / max_total_num_tokens (capacity win)
  - greedy generations (coherence check)
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
# Run the engine core in-process so we can introspect the live model runner.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", required=True)
    ap.add_argument("--model", default="/workspace/models/Qwen/Qwen3-0.6B")
    ap.add_argument("--backend", default="TRITON_ATTN")
    ap.add_argument("--kv-cache-memory", type=int, default=2 * 1024**3)
    ap.add_argument("--gpu-mem-util", type=float, default=0.12)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rec: dict = {"kv_cache_dtype": args.kv_cache_dtype, "backend_req": args.backend}

    def emit():
        print("E50RESULT " + json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                f.write(json.dumps(rec) + "\n")

    try:
        llm = LLM(
            model=args.model,
            kv_cache_dtype=args.kv_cache_dtype,
            attention_backend=(
                args.backend
                if args.backend and args.backend.lower() != "none"
                else None
            ),
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            kv_cache_memory_bytes=args.kv_cache_memory,
            enforce_eager=args.enforce_eager,
            trust_remote_code=True,
            disable_log_stats=True,
        )
    except Exception as e:  # noqa: BLE001
        rec["boot"] = "FAIL"
        rec["error"] = f"{type(e).__name__}: {e}"
        emit()
        return 1

    rec["boot"] = "OK"

    # --- introspect the live engine ------------------------------------
    try:
        mr = (
            llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        )
        kv = mr.kv_caches[0]
        rec["kv_tensor"] = {
            "shape": list(kv.shape),
            "dtype": str(kv.dtype),
            "elem_size": kv.element_size(),
        }
        total_kv_bytes = sum(t.numel() * t.element_size() for t in mr.kv_caches)
        rec["n_kv_tensors"] = len(mr.kv_caches)
        rec["total_kv_bytes"] = total_kv_bytes

        impls = {}
        for (
            layer_name,
            attn,
        ) in mr.vllm_config.compilation_config.static_forward_context.items():
            impl = getattr(attn, "impl", None)
            if impl is None:
                continue
            impls.setdefault(
                type(impl).__name__,
                {
                    "impl_kv_cache_dtype": getattr(impl, "kv_cache_dtype", None),
                    "quant_mode": str(getattr(impl, "_kv_quant_mode", None)),
                    "n_layers": 0,
                },
            )
            impls[type(impl).__name__]["n_layers"] += 1
        rec["attention_impls"] = impls

        cc = mr.vllm_config.cache_config
        rec["num_gpu_blocks"] = cc.num_gpu_blocks
        rec["block_size"] = cc.block_size
        rec["max_total_num_tokens"] = (
            cc.num_gpu_blocks * cc.block_size if cc.num_gpu_blocks else None
        )
        if rec["max_total_num_tokens"]:
            rec["kv_bytes_per_token"] = total_kv_bytes / rec["max_total_num_tokens"]
    except Exception as e:  # noqa: BLE001
        rec["introspect_error"] = f"{type(e).__name__}: {e}"

    # --- coherence ------------------------------------------------------
    prompts = [
        "The capital of France is",
        "Q: What is 17 times 3?\nA:",
        "List the first five prime numbers, comma separated:",
        "Once upon a time in a small village by the sea, there lived a fisherman who",
    ]
    try:
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48))
        rec["generations"] = [o.outputs[0].text for o in outs]
        rec["serve"] = "OK"
    except Exception as e:  # noqa: BLE001
        rec["serve"] = "FAIL"
        rec["serve_error"] = f"{type(e).__name__}: {e}"

    emit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
