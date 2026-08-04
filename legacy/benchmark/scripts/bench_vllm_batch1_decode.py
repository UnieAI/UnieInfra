"""Batch-1 decode throughput for a vLLM engine, one arm per process.

Measures decode-only tok/s the way E33 did: time a 1-token generation (that is
prefill + one decode) and a long generation, and take the difference over the
extra tokens. CUDA graphs are captured at engine construction, so the ONLY
honest way to A/B a kernel is one process per arm -- toggling a flag after
startup replays the same captured graph.
"""

import argparse
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/workspace/models/Qwen/Qwen3-8B")
parser.add_argument("--quantization", default=None)
parser.add_argument("--out-tokens", type=int, default=256)
parser.add_argument("--reps", type=int, default=3)
parser.add_argument("--label", default="arm")
parser.add_argument("--json", default=None)
parser.add_argument("--enforce-eager", action="store_true")
# Batch-1 decode needs the weights plus a tiny KV pool, so the default 0.85 is
# only there to match earlier experiments. Lower it to co-exist on a shared box.
parser.add_argument("--gpu-mem-util", type=float, default=0.85)
args = parser.parse_args()

from vllm import LLM, SamplingParams  # noqa: E402

kwargs = dict(
    model=args.model,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=args.gpu_mem_util,
    max_num_seqs=1,
    enforce_eager=args.enforce_eager,
    trust_remote_code=True,
    disable_log_stats=True,
)
if args.quantization:
    kwargs["quantization"] = args.quantization

llm = LLM(**kwargs)
prompt = "Explain in detail how a modern GPU executes a matrix multiplication."


def run(max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sp, use_tqdm=False)
    return time.perf_counter() - t0, outs[0].outputs[0]


# warm up (also forces graph replay paths to be hot)
run(8)

t1, _ = run(1)
samples = []
for _ in range(args.reps):
    t_short, _ = run(1)
    t_long, out = run(args.out_tokens)
    n = len(out.token_ids)
    tps = (n - 1) / max(t_long - t_short, 1e-9)
    samples.append(tps)

_, out = run(64)
text = out.text

result = {
    "label": args.label,
    "quantization": args.quantization,
    "enforce_eager": args.enforce_eager,
    "decode_tok_s_samples": samples,
    "decode_tok_s_best": max(samples),
    "decode_tok_s_median": sorted(samples)[len(samples) // 2],
    "sample_text": text[:400],
}

# Selection evidence, when our plugin is live.
try:
    from unieinfra_kernel.dispatch import selection_log
    from unieinfra_kernel.ops.gemm.decode_gemv_triton import gemv_stats

    result["selection_log"] = selection_log()
    result["gemv_stats"] = gemv_stats()
except Exception as exc:  # noqa: BLE001
    result["selection_log_error"] = repr(exc)

print("UNIEINFRA_RESULT " + json.dumps(result))
if args.json:
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=2)
