#!/usr/bin/env python3
"""Light steady-state decode profile: "know what to fix next" in one command.

Answers one question after a sweep: at batch-1 (latency-bound) and batch-128
(throughput-bound) decode, which kernels actually eat the time, and what's the
first lever to pull?

Two capture modes, both always-available (no root / CAP_SYS_ADMIN needed --
`ncu` hardware counters are blocked in this environment with
RmProfilingAdminOnly, so this script deliberately does NOT use ncu):

  --mode torch (default)
      Drives the ALREADY-RUNNING server (the same one benchmark/run_sweep.sh
      just hit) via its built-in profiler control endpoints -- see
      python/tokenspeed/runtime/entrypoints/control_server.py
      (`/start_profile`, `/stop_profile` on `main_port + 1`, proxied to a
      torch.profiler.profile(...) capture in the scheduler process; see
      python/tokenspeed/runtime/engine/request_handler.py). This profiles the
      REAL serving path -- prefix cache, continuous batching, everything --
      not a synthetic offline harness. Output: a Chrome trace
      `<profile_id>-TP0.trace.json.gz` per batch size, pulled from the
      container with `docker cp` if the server runs in one.

  --mode nsys
      Wraps a short-lived in-process decode harness (adapted from
      benchmark/profiles/_scratch/ncu_decode_harness.py, which already proved
      this pattern for the ncu investigation) under `nsys profile`, then
      reads back `nsys stats --report cuda_gpu_kern_sum`. Useful when you want
      a fully isolated capture (no server, no concurrent traffic) or when the
      server's own profiler endpoints aren't reachable.

Both modes feed the SAME hot-kernel aggregation + categorizer + "likely
lever" heuristic, so the two are comparable.

Richer option when unlocked: if CAP_SYS_ADMIN / RmProfilingAdminOnly ever gets
lifted on this box, prefer `ncu` for hardware-counter-backed roofline analysis
(achieved vs peak DRAM bandwidth, tensor-core utilization, occupancy) -- see
the `ncu-report-skill` skill and benchmark/profiles/_scratch/ncu_decode_harness.py
for that path. This script's "likely lever" hints are time-share heuristics,
not a roofline -- confirm any lever with ncu before spending real
optimization effort on it.

This script NEEDS A GPU (a running server for --mode torch, or CUDA locally
for --mode nsys) TO ACTUALLY RUN. It is wired here but not executed as part
of building this harness -- see benchmark/README.md hygiene rules.

Usage:
    profile_hot_decode.py --label clean --base-url http://127.0.0.1:8000 \\
        --model bench-model --container unieinfra-dev

    profile_hot_decode.py --label clean --mode nsys --model /path/to/model

Python stdlib only (urllib, json, gzip, subprocess, concurrent.futures) -- no
heavy deps.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_SIZES = [1, 128]
DEFAULT_MAX_TOKENS = 64
DEFAULT_TOP_N = 15


def eprint(*a, **kw):
    print(*a, file=sys.stderr, **kw)


# ---------------------------------------------------------------------------
# Kernel-name categorizer -- shared by both capture modes. Consistent with
# (but not a copy-paste of, since the source formats differ) the categories
# used in benchmark/nsys_out/analyze_steps.py's steady-state gemv/attention
# breakdown, so results from that earlier ncu/nsys investigation and this
# tool's output read the same way.
# ---------------------------------------------------------------------------
def categorize(name: str) -> str:
    n = name.lower()
    if "gemvx" in n or "gemv" in n:
        return "gemv"
    if "gemm" in n or "cutlass" in n and "wmma" in n or "cutlass_" in n:
        return "gemm"
    if (
        "batchprefillwithpagedkv" in n
        or "batchdecodewithpagedkv" in n
        or "flashattn" in n
        or "flash_attn" in n
    ):
        return "attention"
    if "rmsnorm" in n:
        return "rmsnorm"
    if "rotary" in n or "rope" in n:
        return "rope"
    if "act_and_mul" in n or "silu" in n:
        return "silu"
    if "store_kv" in n or "kv_cache" in n or "setkv" in n:
        return "kv_store"
    if "memcpy" in n or "memset" in n:
        return "memcpy"
    return "other_glue"


def likely_lever(cat_pct: dict, batch_size: int) -> str:
    """One-line heuristic hint. This is a time-share heuristic, NOT a
    roofline -- confirm with ncu before committing real effort (see module
    docstring)."""
    if not cat_pct:
        return "no kernel data captured -- check the profile actually ran (server up? workload reached steady state?)."
    dominant = max(cat_pct, key=cat_pct.get)
    pct = cat_pct[dominant]
    if dominant in ("gemv",) and pct > 50:
        if batch_size <= 1:
            return (
                f"gemv-dominated ({pct:.0f}%) at batch={batch_size}: single-token decode is DRAM-bandwidth "
                "bound on weight reads. Lever = weight quantization (fp8/int4) to cut bytes/token, or raise "
                "effective batch size (continuous batching) so more tokens amortize each weight read."
            )
        return (
            f"gemv-dominated ({pct:.0f}%) at batch={batch_size}: decode is still memory-bound even at this "
            "batch size -- check max_num_seqs / cuda-graph capture size are actually letting requests batch. "
            "Lever = raise batch/continuous-batching headroom before reaching for compute-side fixes."
        )
    if dominant == "gemm" and pct > 50:
        return (
            f"gemm-dominated ({pct:.0f}%) at batch={batch_size}: compute-bound GEMMs. Lever = GEMM kernel "
            "selection/autotune (cutlass/cuBLAS) or tensor-core utilization; weight quantization helps less "
            "here than in the gemv-bound (low-batch) regime."
        )
    if dominant == "attention" and pct > 25:
        return (
            f"attention is the largest single category ({pct:.0f}%) at batch={batch_size}. Lever = faster "
            "attention kernel (flashinfer/FA2/FA3 backend swap) or KV-cache quantization to cut memory "
            "traffic per step."
        )
    if dominant == "other_glue" and pct > 30:
        return (
            f"launch/glue overhead dominates ({pct:.0f}% in many small elementwise/index/copy kernels). "
            "Lever = CUDA graphs (if not already on) or operator fusion to cut per-step host-side dispatch."
        )
    if dominant == "memcpy" and pct > 10:
        return (
            f"memcpy/memset is an unusually large share ({pct:.0f}%). Lever = check for avoidable "
            "host<->device copies or KV-cache layout churn."
        )
    return (
        f"no single dominant category (top = {dominant} at {pct:.0f}%, <~50%): decode looks reasonably "
        "balanced at this batch size. Re-check at the OTHER batch size, or unlock ncu for a roofline "
        "(achieved vs peak DRAM bandwidth / tensor-core occupancy) to find the next lever."
    )


def aggregate(kernel_durations: dict) -> tuple[list, dict]:
    """kernel_durations: {name: total_duration_ns}. Returns (sorted [(name,dur)], {cat: pct})."""
    total = sum(kernel_durations.values())
    ranked = sorted(kernel_durations.items(), key=lambda kv: -kv[1])
    cat_dur: dict = {}
    for name, dur in kernel_durations.items():
        c = categorize(name)
        cat_dur[c] = cat_dur.get(c, 0) + dur
    cat_pct = {c: (d / total * 100.0 if total else 0.0) for c, d in cat_dur.items()}
    return ranked, cat_pct


def render_report(
    label: str,
    mode: str,
    batch_size: int,
    ranked: list,
    cat_pct: dict,
    total_ns: int,
    top_n: int,
) -> str:
    lines = [f"# Hot decode kernels: {label} bs={batch_size} (mode={mode})\n"]
    if total_ns <= 0:
        lines.append("_(no kernel time captured)_\n")
        return "\n".join(lines)
    lines.append(f"Total captured GPU-kernel busy time: {total_ns/1e3:.1f} us\n")
    lines.append("## Category breakdown\n")
    lines.append("| category | % of GPU busy |")
    lines.append("|---|---|")
    for c, p in sorted(cat_pct.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {c} | {p:.1f}% |")
    lines.append("")
    lines.append(f"## Top {top_n} kernels\n")
    lines.append("| # | duration share | duration (us) | category | kernel |")
    lines.append("|---|---|---|---|---|")
    for i, (name, dur) in enumerate(ranked[:top_n], 1):
        pct = dur / total_ns * 100.0 if total_ns else 0.0
        short = name if len(name) <= 100 else name[:97] + "..."
        lines.append(
            f"| {i} | {pct:.1f}% | {dur/1e3:.2f} | {categorize(name)} | `{short}` |"
        )
    lines.append("")
    lines.append(f"## Likely lever\n\n{likely_lever(cat_pct, batch_size)}\n")
    lines.append(
        "_(heuristic from kernel time-share, not a hardware-counter roofline -- confirm with ncu "
        "when unlocked; see module docstring.)_\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# mode=torch: drive the already-running server's /start_profile,/stop_profile
# ---------------------------------------------------------------------------
def http_post_json(url: str, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"raw": raw.decode(errors="replace")}


def drive_workload(
    base_url: str, model: str, batch_size: int, max_tokens: int, prompt: str
):
    """Fire `batch_size` concurrent completions requests and wait for all of them.
    This is what gets captured between start_profile/stop_profile."""
    url = base_url.rstrip("/") + "/v1/completions"

    def one():
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        return http_post_json(url, body, timeout=120.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(batch_size, 1)) as ex:
        futs = [ex.submit(one) for _ in range(batch_size)]
        for f in concurrent.futures.as_completed(futs):
            f.result()  # surface exceptions


def docker_cp_out(container: str, remote_path: str, local_dir: Path) -> Path | None:
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / Path(remote_path).name
    try:
        subprocess.run(
            ["docker", "cp", f"{container}:{remote_path}", str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        return dest
    except subprocess.CalledProcessError as e:
        eprint(
            f"[profile_hot_decode] docker cp failed for {remote_path}: {e.stderr.strip()}"
        )
        return None


def find_trace_in_container(
    container: str, output_dir: str, profile_id: str
) -> str | None:
    try:
        out = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "sh",
                "-c",
                f"ls {output_dir}/{profile_id}*.trace.json.gz 2>/dev/null",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        eprint(
            "[profile_hot_decode] `docker` not found on PATH -- pass --container '' if the server "
            "is not containerized (output_dir must then be host-visible)."
        )
        return None
    paths = [p for p in out.stdout.splitlines() if p.strip()]
    return paths[0] if paths else None


def parse_chrome_trace(path: Path) -> dict:
    """Return {kernel_name: total_duration_ns} for device-side kernel events."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    durations: dict = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur_us = e.get("dur")
        if dur_us is None:
            continue
        name = e.get("name", "?")
        durations[name] = (
            durations.get(name, 0) + dur_us * 1000.0
        )  # trace "dur" is us -> ns
    return durations


def run_mode_torch(args, batch_size: int) -> dict | None:
    control_port = args.control_port or (args.port + 1)
    control_url = (
        f"http://127.0.0.1:{control_port}"
        if args.control_host is None
        else f"http://{args.control_host}:{control_port}"
    )
    profile_id = f"{args.label}_bs{batch_size}"
    output_dir = f"{args.remote_output_dir.rstrip('/')}/{args.label}"

    prompt = "Explain in detail what makes a good distributed inference engine. " * 8

    print(f"[profile_hot_decode] bs={batch_size}: warmup request ...")
    try:
        drive_workload(args.base_url, args.model, min(batch_size, 4), 8, prompt)
    except (urllib.error.URLError, OSError) as e:
        eprint(
            f"[profile_hot_decode] warmup request failed ({e}); is the server up at {args.base_url}?"
        )
        return None

    print(
        f"[profile_hot_decode] bs={batch_size}: /start_profile (profile_id={profile_id}, output_dir={output_dir}) ..."
    )
    try:
        http_post_json(
            f"{control_url}/start_profile",
            {
                "output_dir": output_dir,
                "activities": ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": False,
                "profile_id": profile_id,
            },
        )
    except (urllib.error.URLError, OSError) as e:
        eprint(
            f"[profile_hot_decode] /start_profile failed at {control_url} ({e}); "
            f"is the control server up (main_port+1, or --control-port)?"
        )
        return None

    try:
        drive_workload(args.base_url, args.model, batch_size, args.max_tokens, prompt)
    finally:
        print(f"[profile_hot_decode] bs={batch_size}: /stop_profile ...")
        try:
            http_post_json(f"{control_url}/stop_profile", {})
        except (urllib.error.URLError, OSError) as e:
            eprint(
                f"[profile_hot_decode] /stop_profile failed ({e}) -- trace may be incomplete/missing"
            )

    if args.container:
        remote_trace = find_trace_in_container(args.container, output_dir, profile_id)
        if not remote_trace:
            eprint(
                f"[profile_hot_decode] no trace file found in {args.container}:{output_dir} "
                f"for profile_id={profile_id}"
            )
            return None
        local_trace = docker_cp_out(
            args.container, remote_trace, args.out_dir / args.label
        )
        if not local_trace:
            return None
    else:
        local_trace = Path(output_dir) / f"{profile_id}-TP0.trace.json.gz"
        if not local_trace.exists():
            eprint(
                f"[profile_hot_decode] expected trace at {local_trace} (no --container given, "
                f"so assuming it's host-visible) but it does not exist"
            )
            return None

    return parse_chrome_trace(local_trace)


# ---------------------------------------------------------------------------
# mode=nsys: short-lived in-process harness under `nsys profile`
# ---------------------------------------------------------------------------
NSYS_HARNESS_TEMPLATE = """
import os
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
from tokenspeed.runtime.entrypoints.engine import Engine

MODEL = {model!r}
BS = {batch_size!r}
MAX_NEW = {max_tokens!r}

eng = Engine(
    model=MODEL, dtype="bfloat16", world_size=1, ep_size=1, trust_remote_code=True,
    enforce_eager=True, disable_custom_all_reduce=True, attention_backend="triton",
    gpu_memory_utilization=0.85, max_model_len=4096,
    max_num_seqs=max(BS, 256) if BS > 1 else 32, log_level="error",
)
prompts = "The capital of France is" if BS == 1 else [
    f"Question {{i}}: Explain in one paragraph what makes a good API." for i in range(BS)
]
eng.generate(prompt=prompts, sampling_params={{"temperature": 0.0, "max_new_tokens": 1}})  # warmup
eng.generate(prompt=prompts, sampling_params={{"temperature": 0.0, "max_new_tokens": MAX_NEW}})  # measured
eng.shutdown()
"""


def run_mode_nsys(args, batch_size: int) -> dict | None:
    if shutil.which("nsys") is None:
        eprint(
            "[profile_hot_decode] `nsys` not found on PATH -- install Nsight Systems or use --mode torch"
        )
        return None
    out_dir = args.out_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    harness = out_dir / f"_nsys_harness_bs{batch_size}.py"
    harness.write_text(
        NSYS_HARNESS_TEMPLATE.format(
            model=args.model, batch_size=batch_size, max_tokens=args.max_tokens
        )
    )
    rep = out_dir / f"{args.label}_bs{batch_size}"
    print(f"[profile_hot_decode] bs={batch_size}: nsys profile -> {rep}.nsys-rep ...")
    try:
        subprocess.run(
            [
                "nsys",
                "profile",
                "-t",
                "cuda,nvtx",
                "-o",
                str(rep),
                "--force-overwrite=true",
                sys.executable,
                str(harness),
            ],
            check=True,
        )
        subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--format",
                "csv",
                "--output",
                str(rep),
                f"{rep}.nsys-rep",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        eprint(f"[profile_hot_decode] nsys invocation failed: {e}")
        return None
    kern_csv = Path(f"{rep}_cuda_gpu_kern_sum.csv")
    if not kern_csv.exists():
        eprint(
            f"[profile_hot_decode] expected {kern_csv} after `nsys stats` but it's missing"
        )
        return None
    import csv as csv_mod

    durations: dict = {}
    with open(kern_csv, newline="") as f:
        for row in csv_mod.DictReader(f):
            name = row.get("Name", "?")
            try:
                durations[name] = durations.get(name, 0) + float(
                    row.get("Total Time (ns)", 0)
                )
            except ValueError:
                continue
    return durations


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--label",
        required=True,
        help="run label, e.g. matches results/<engine>/<label>",
    )
    ap.add_argument("--engine", default="unieinfra")
    ap.add_argument("--mode", choices=["torch", "nsys"], default="torch")
    ap.add_argument(
        "--base-url",
        default=None,
        help="server base URL for --mode torch (default: " "http://127.0.0.1:<port>)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8000,
        help="main server port (used to derive --base-url "
        "and, as port+1, the default control port)",
    )
    ap.add_argument(
        "--control-port",
        type=int,
        default=None,
        help="override control port (default: port+1)",
    )
    ap.add_argument(
        "--control-host",
        default=None,
        help="override control host (default: same host as --base-url)",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="model path (--mode nsys) or served name (--mode torch, "
        "should match the /v1/completions `model` field, typically 'bench-model')",
    )
    ap.add_argument(
        "--container",
        default="",
        help="docker container the server runs in, for docker cp "
        "of the trace file (--mode torch only); empty = assume host-visible paths",
    )
    ap.add_argument(
        "--remote-output-dir",
        default="/tmp/profile_hot_decode",
        help="directory (inside --container, if set) the server writes trace files to",
    )
    ap.add_argument(
        "--batch-sizes", default=",".join(str(b) for b in DEFAULT_BATCH_SIZES)
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="decode steps to capture per request (default: %(default)s)",
    )
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--out-dir", type=Path, default=BENCH_DIR / "profiles")
    args = ap.parse_args()

    args.base_url = args.base_url or f"http://127.0.0.1:{args.port}"
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    any_ok = False
    for bs in batch_sizes:
        t0 = time.time()
        if args.mode == "torch":
            durations = run_mode_torch(args, bs)
        else:
            durations = run_mode_nsys(args, bs)
        if durations is None:
            eprint(
                f"[profile_hot_decode] bs={bs}: no data captured (see warnings above) -- skipping report"
            )
            continue
        any_ok = True
        ranked, cat_pct = aggregate(durations)
        total_ns = sum(durations.values())
        report = render_report(
            args.label, args.mode, bs, ranked, cat_pct, total_ns, args.top_n
        )
        out_dir = args.out_dir / args.label
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"bs{bs}_hotkernels.md"
        report_path.write_text(report)
        print(
            f"[profile_hot_decode] bs={bs}: wrote {report_path}  ({time.time()-t0:.1f}s)"
        )
        print(f"[profile_hot_decode] bs={bs}: {likely_lever(cat_pct, bs)}")

    if not any_ok:
        eprint(
            "[profile_hot_decode] no batch size produced data -- this needs a live server "
            "(--mode torch) or a local GPU (--mode nsys); nothing to report."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
