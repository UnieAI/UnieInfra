"""Export AgentX results on 4x MI350P as dashboard rows (JSON array).

One row per (engine, concurrency): the latest successful run of that cell.
Runs whose directory starts with FAILED-, smoke runs and runs without an
AiPerf export are skipped.

Usage: python3 export_dashboard.py OUT.json
"""

import glob
import json
import os
import re
import sys

ROOT = "/home/smc/workspace/roy/results/agentx"
GPUS = 4

# Directory prefix -> engine label shown on the dashboard.
ENGINES = [
    # Stress runs (think time removed, unofficial) are their own series.
    ("unieinfra-stress", "UnieInfra optimized (stress)"),
    ("unieinfra-port-", "UnieInfra ported (stress)"),
    ("unieinfra-portofficial", "UnieInfra ported"),
    ("unieinfra-mix-", "UnieInfra mixed batch (stress)"),
    ("unieinfra-mixbig", "UnieInfra mixed batch 16K (stress)"),
    ("unieinfra-lpm-", "UnieInfra cache-aware (stress)"),
    ("unieinfra-lpmseq", "UnieInfra cache-aware seqs-capped (stress)"),
    ("unieinfra-iris32-", "UnieInfra Iris32 (stress)"),
    ("unieinfra-k62", "UnieInfra kernels+sched (stress)"),
    ("unieinfra-k67", "UnieInfra +attn (stress)"),
    ("unieinfra-k69", "UnieInfra +tail32 (stress)"),
    ("unieinfra-k75", "UnieInfra +block32 GEMM (stress)"),
    ("unieinfra-k77", "UnieInfra +no-fill (stress)"),
    ("unieinfra-k79", "UnieInfra +KV 33.5M tokens (stress)"),
    ("unieinfra-k80b", "UnieInfra verify 3 drafts (stress)"),
    ("unieinfra-k92", "UnieInfra +INT8 prefill all-reduce (stress)"),
    ("unieinfra-mix94", "UnieInfra +INT8, mixed batch (stress)"),
    ("unieinfra-chunk8192", "UnieInfra +INT8, 8K chunks (stress)"),
    ("unieinfra-chunk4096", "UnieInfra +INT8, 4K chunks (stress)"),
    ("unieinfra-chunk2048", "UnieInfra +INT8, 2K chunks (stress)"),
    ("unieinfra-samp88", "UnieInfra sampled, k79 build (stress)"),
    ("unieinfra-samp91", "UnieInfra sampled, 4K + INT8 (stress)"),
    ("unieinfra-cpl90", "UnieInfra coupled draft, sampled (stress)"),
    ("unieinfra-best1", "UnieInfra coupled draft, sampled (stress)"),
    ("unieinfra-k100", "UnieInfra sampled, INT8 + 4K chunks (stress)"),
    ("unieinfra-SBP", "BvB stress: UnieInfra + block verify"),
    ("unieinfra-STB", "BvB stress: UnieInfra + block verify + prefill TBO"),
    ("unieinfra-TBO2T1", "BvB stress: UnieInfra + block verify + prefill TBO"),
    ("unieinfra-TBO2T0", "TBO A/B off (stress)"),
    ("unieinfra-RPG", "Repeat A/B: gluon MoE + block verify (stress)"),
    ("unieinfra-RPA", "Repeat A/B: aiter MoE + block verify (stress)"),
    ("unieinfra-SBH", "BvB stress: UnieInfra + block verify + aiter MoE + host fix"),
    ("unieinfra-SBF", "BvB stress: UnieInfra + block verify + aiter MoE"),
    ("unieinfra-SBA", "BvB stress: UnieInfra + block verify, aiter MoE + frag fix"),
    ("unieinfra-SBV", "UnieInfra + block verify, aiter MoE (stress)"),
    ("unieinfra-abt2", "AB tail2"),
    ("unieinfra-abt32", "AB tail32"),
    ("vllm-stressbig", "vLLM (stress, 2× max-num-seqs)"),
    ("vllm-stress", "vLLM (stress)"),
    ("unieinfra-final", "UnieInfra optimized"),
    ("unieinfra-comm", "UnieInfra optimized"),
    ("unieinfra-iter", "UnieInfra optimized"),
    ("unieinfra-unieinfra-opt", "UnieInfra optimized"),
    ("unieinfra-unieinfra-nooverlap", "UnieInfra overlap off"),
    ("unieinfra-baseline", "UnieInfra baseline"),
    ("vllm-", "vLLM"),
    ("sglang-", "SGLang"),
]


def engine_of(name: str) -> str | None:
    if name.startswith("FAILED") or "-smoke-" in name or "FAILED" in name:
        return None
    m = re.match(r"vllm-stressseq(\d+)-", name)
    if m:
        return f"vLLM (stress, max-num-seqs {m.group(1)})"
    # Best-vs-best sweeps (Nov 12): BV = official (recorded gaps, 900 s), SB = stress (600 s).
    # kind r = vLLM recipe cap (CONC/4), x = 2*CONC cap; trailing k = UnieInfra KV token cap lifted.
    m = re.match(r"(unieinfra|vllm)-(BV|SB)\d+([rx])[uv](k?)-", name)
    if m:
        eng = "UnieInfra" if m.group(1) == "unieinfra" else "vLLM"
        mode = "official" if m.group(2) == "BV" else "stress"
        cap = "recipe cap" if m.group(3) == "r" else "2x cap"
        return f"BvB {mode}: {eng} {cap}" + (", KV uncapped" if m.group(4) else "")
    m = re.match(r"unieinfra-parS(\d+)-", name)
    if m:
        # The pareto sweep has three running-request caps per concurrency, and rows
        # are keyed by (engine, concurrency): a shared label would keep only one cap.
        return f"UnieInfra pareto, cap{m.group(1)} (official)"
    for prefix, label in ENGINES:
        if name.startswith(prefix):
            return label
    return None


def stat(j, key, s):
    v = j.get(key, {}).get(s)
    return None if v is None else float(v)


def main() -> None:
    cells = {}
    for d in sorted(glob.glob(f"{ROOT}/*")):
        name = os.path.basename(d)
        engine = engine_of(name)
        f = os.path.join(d, "aiperf_artifacts/profile_export_aiperf.json")
        m = re.search(r"-c(\d+)-", name)
        if engine is None or m is None or not os.path.exists(f):
            continue
        j = json.load(open(f))
        conc = int(m.group(1))
        stamp = name.rsplit("-", 1)[-1]
        out_tps = stat(j, "output_token_throughput", "avg")
        row = {
            "cell": f"{engine} c{conc}",
            "engine": engine,
            "concurrency": conc,
            "duration_s": round(stat(j, "benchmark_duration", "avg"), 1),
            "requests": int(stat(j, "request_count", "avg")),
            "rpm": round(stat(j, "request_throughput", "avg") * 60, 2),
            "output_tps": round(out_tps, 1),
            "output_tps_per_gpu": round(out_tps / GPUS, 1),
            "input_tps": round(stat(j, "input_token_throughput", "avg")),
            "total_tps_per_gpu": round((stat(j, "input_token_throughput", "avg") + out_tps) / GPUS),
            "ttft_mean_ms": round(stat(j, "time_to_first_token", "avg")),
            "ttft_p50_ms": round(stat(j, "time_to_first_token", "p50")),
            "ttft_p99_ms": round(stat(j, "time_to_first_token", "p99")),
            "itl_p50_ms": round(stat(j, "inter_token_latency", "p50"), 2),
            "itl_p99_ms": round(stat(j, "inter_token_latency", "p99"), 2),
            "tps_user_p50": round(stat(j, "output_token_throughput_per_user", "p50"), 1),
            "err_pct": round(stat(j, "request_error_rate", "avg") or 0.0, 2),
            "run": name,
        }
        key = (engine, conc)
        if key not in cells or stamp > cells[key][0]:
            cells[key] = (stamp, row)
    rows = sorted((r for _, r in cells.values()), key=lambda r: (r["engine"], r["concurrency"]))
    json.dump(rows, open(sys.argv[1], "w"), indent=1)
    for r in rows:
        print(f'{r["cell"]:28s} rpm={r["rpm"]:6.1f} out={r["output_tps"]:7.1f} ttft_mean={r["ttft_mean_ms"]:6d} tps_user={r["tps_user_p50"]:6.1f}')


if __name__ == "__main__":
    main()
