#!/usr/bin/env python3
"""Kernel-level benchmark + correctness harness for tokenspeed-kernel.

This is the kernel-level counterpart of ``eval/steady_bench.py``. That one
measures a server; this one measures a single registered operator
implementation, at one shape, in isolation. The reason both exist is that the
three retractions this project is named after (see ``../README.md``) were all
*measurement* failures, not idea failures, and a kernel harness has its own set
of ways to be confidently wrong. Each guard below names the one it blocks.

WHAT IT DOES

Given an operator ``(family, mode)`` and a shape, it:

  1. Enumerates every kernel the platform admits for it, straight out of
     ``KernelRegistry.get()._by_operator`` -> ``spec.capability.satisfied_by()``
     -> format-signature -> trait filtering. Enumeration is reported in stages
     so "this kernel never ran" is distinguishable from "this kernel lost".
  2. Runs each candidate through the *public dispatch function* with
     ``override=<kernel name>``, so argument marshalling, online quantization
     and profiling scopes are exercised exactly as in production. A candidate
     that raises is recorded with its exact exception type and message; a
     raising candidate is a result, not a gap in the table.
  3. Compares every candidate against ONE explicitly named reference, reporting
     max abs diff, mean abs diff, relative L2 and cosine similarity.
  4. Times with CUDA events, after warmup, over many samples, and reports the
     median with its spread -- and REFUSES to report a number whose sample
     series was not stationary.
  5. Emits JSON (diffable across agents and runs) plus a readable table.

FAILURE MODES GUARDED, AND WHY EACH IS HERE

  * ``implicit reference`` -- gemma-4 retraction: correctness was judged against
    what the output was expected to look like. Every numerics row here names
    the reference it was compared against, in the JSON and in the table. There
    is no default reference: a case that does not declare one does not run.
    Note that for several operators on sm_120 there is exactly ONE registered
    kernel, so "compare the candidates to each other" would be vacuous -- the
    references here are independent in-harness fp32 torch implementations
    (``harness:*``) unless a REFERENCE-band kernel exists in the registry.

  * ``cosine is blind to scale`` -- a kernel with the wrong ``softmax_scale`` or
    a missing alpha has cosine ~1.000000 and is completely wrong. ``rel_l2`` is
    reported alongside and is the number to read first.

  * ``single timing`` -- one call, or one averaged loop, cannot show whether the
    GPU was still ramping. N samples, median, p10/p90 spread, full series kept
    in the JSON.

  * ``clock ramp`` -- the first samples on a cold card are slow, and a run that
    stops early records the ramp as the answer. The sample series is put
    through ``kda.gates.gate_stationary`` (the same gate the serving harness
    uses, whose origin was a 13-sigma result measured against an arm that had
    not converged). A non-stationary series is re-run once with a longer
    warmup; if it is still moving, the latency and throughput fields are
    ``null`` and ``rejected`` names the reason. The observed numbers stay in
    ``observed`` for diagnosis but are not reportable.

  * ``nvidia-smi utilisation`` -- BROKEN on GPUs 2 and 3 of this machine: it
    reports sm=100% on a completely idle card (see
    ``experiment/kda-sm120/docs/log.md`` entry E22). This harness never reads
    utilisation, and no conclusion here rests on it. What it does read is SM
    clock, temperature, power and the *process list*, all of which E22 found
    consistent with reality.

  * ``shared card`` -- GPU 3 is shared with other agents. A neighbour's kernel
    inflates individual samples at random. The median plus p10/p90 spread makes
    that visible instead of averaging it in, other compute processes are
    counted at start and end via NVML, and a run whose process count changed
    mid-flight is flagged. Contention is *not* silently corrected for.

  * ``input clobbering`` -- a kernel that writes through one of its inputs
    corrupts every candidate measured after it, and the corruption shows up as
    a numerics failure attributed to the wrong kernel. Every input tensor is
    byte-checksummed before and after each candidate.

  * ``throughput in the wrong unit`` -- a memory-bound op quoted in TFLOP/s
    looks terrible and a compute-bound op quoted in GB/s looks fantastic, and
    neither number means anything. Each case declares its natural unit, and
    both TFLOP/s and GB/s are reported with the primary one marked.

  * ``FLOPs that do not match the mask`` -- causal attention does about half the
    work of the dense pair count. FLOPs here are computed from the *attended
    pair count* implied by the actual mask, per sequence, not from a shape
    formula.

  * ``unattributed number`` -- ``kda.gates.gate_attribution``: every result
    carries its model geometry label, shape dict and kernel name.

NOT GUARDED (known gaps, stated so nobody assumes otherwise)

  * Per-kernel *autotuning* state. Triton kernels cache their autotune choice
    per process. Candidates are measured in one process in a fixed order; a
    kernel measured second may benefit from a warmer allocator. Warmup is
    per-candidate, but the caching allocator is not reset between them.
  * Power/thermal steady state across a long sweep. Each case is short by
    design (shared card), so a long sweep can drift between the first case and
    the last. Clocks are recorded per candidate so this is visible, not fixed.
  * Cross-run comparability of absolute numbers when the card is shared with a
    different neighbour. Compare within a run; across runs compare only the
    ordering, and check ``gpu.other_process_count``.
  * fp8/fp4 numerics tolerance. The harness reports the diffs; it does not know
    what tolerance is acceptable for a quantized path. That judgement belongs
    to whoever owns the kernel.

USAGE

    docker run --rm --gpus '"device=3"' -e FLASHINFER_DISABLE_VERSION_CHECK=1 \\
      -v /workspace/roy/UnieInfra/experiment/unieai-kda:/probe:ro \\
      tokenspeed:sm120-dev python3 /probe/eval/kernel_bench.py --list

    ... --json /out/baseline.json --cases mha_decode_qwen35_27b_b32_ctx4096

Out-of-tree plugins: pass ``--plugin tokenspeed_kernel_myplugin`` (repeatable)
or set ``KB_PLUGINS=mod1,mod2``. The module is imported before enumeration, so
anything it registers appears as a candidate with no change to this file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform as _platform
import statistics as st
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# --- kda gates -------------------------------------------------------------
# The stationarity gate is shared with the serving harness on purpose: the
# failure it blocks (reporting the mean of a series that was still moving) is
# identical at both levels, and a second copy of it would drift.
_KDA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KDA_ROOT not in sys.path:
    sys.path.insert(0, _KDA_ROOT)
try:
    from kda.gates import GateResult, evaluate_gates, gate_attribution, gate_stationary

    _HAVE_GATES = True
except Exception:  # pragma: no cover - harness must still run standalone
    _HAVE_GATES = False

    @dataclass
    class GateResult:  # type: ignore[no-redef]
        name: str
        ok: bool
        detail: str
        values: dict = field(default_factory=dict)

        def __str__(self) -> str:
            return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


HARNESS_VERSION = "kernel_bench/1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Samples, not iterations. Each sample is itself an inner loop timed by one
# pair of CUDA events, sized so the loop lasts ~TARGET_REP_MS; that amortises
# launch overhead without hiding it, and gives a series long enough for the
# stationarity gate (which needs >= 6).
SAMPLES = int(os.environ.get("KB_SAMPLES", "15"))
WARMUP_ITERS = int(os.environ.get("KB_WARMUP", "25"))
TARGET_REP_MS = float(os.environ.get("KB_TARGET_REP_MS", "3.0"))
MAX_INNER = int(os.environ.get("KB_MAX_INNER", "200"))

# Tighter than the serving harness's 5%. Kernel timings have far lower variance
# than serving throughput, and 3% of drift is already enough to reorder two
# candidates -- which is the decision this harness exists to support.
DRIFT_TOL = float(os.environ.get("KB_DRIFT_TOL", "0.03"))
ENVELOPE_TOL = float(os.environ.get("KB_ENVELOPE_TOL", "0.08"))
SWING_FLOOR = float(os.environ.get("KB_SWING_FLOOR", "0.10"))

SEED = int(os.environ.get("KB_SEED", "1234"))
DEV = "cuda"


# ---------------------------------------------------------------------------
# Device facts -- clocks, power, processes. Never utilisation (E22).
# ---------------------------------------------------------------------------


class _NVML:
    """Thin NVML wrapper. Absent NVML is reported, never silently treated as
    healthy -- a missing instrument that reads as a pass is exactly how the
    counter_agreement failure survived six configurations."""

    def __init__(self) -> None:
        self.handle = None
        self.err: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._n = pynvml
            # torch's device 0 is whatever --gpus exposed; map via PCI bus id so
            # the NVML handle is the same physical card torch is using. torch
            # returns the components as ints, so the canonical NVML string has
            # to be rebuilt rather than read off a property.
            p = torch.cuda.get_device_properties(0)
            bus_id = "%08X:%02X:%02X.0" % (
                getattr(p, "pci_domain_id", 0) or 0,
                getattr(p, "pci_bus_id", 0) or 0,
                getattr(p, "pci_device_id", 0) or 0,
            )
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByPciBusId(bus_id.encode())
            except Exception:
                # Under `--gpus '"device=N"'` NVML inside the container usually
                # enumerates only the exposed card, so index 0 is that card.
                # Record the fallback so a wrong-card reading is traceable.
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.err = f"pci lookup {bus_id} failed; fell back to NVML index 0"
        except Exception as exc:
            self.err = f"{type(exc).__name__}: {exc}"

    def snapshot(self) -> dict:
        """Clock / thermal / process facts. ``utilisation is deliberately absent``:
        on GPUs 2 and 3 of this machine it reads 100% on an idle card (E22)."""
        if self.handle is None:
            return {"available": False, "error": self.err,
                    "note": "utilisation is never read; see log.md E22"}
        n = self._n
        out: dict[str, Any] = {"available": True,
                               "note": "utilisation is never read; see log.md E22"}

        def _try(key, fn):
            try:
                out[key] = fn()
            except Exception as exc:
                out[key] = None
                out.setdefault("errors", {})[key] = type(exc).__name__

        _try("sm_clock_mhz", lambda: n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_SM))
        _try("mem_clock_mhz", lambda: n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_MEM))
        _try("max_sm_clock_mhz",
             lambda: n.nvmlDeviceGetMaxClockInfo(self.handle, n.NVML_CLOCK_SM))
        _try("temperature_c",
             lambda: n.nvmlDeviceGetTemperature(self.handle, n.NVML_TEMPERATURE_GPU))
        _try("power_w", lambda: round(n.nvmlDeviceGetPowerUsage(self.handle) / 1000.0, 1))
        _try("throttle_reasons",
             lambda: int(n.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)))
        _try("other_process_count",
             lambda: len(n.nvmlDeviceGetComputeRunningProcesses(self.handle)))
        return out


def l2_bytes() -> int:
    return int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 0)


# Free-card reference for the calibration anchor below: log.md E22 measured a
# bf16 8192^3 GEMM at 427.1 TFLOP/s peak on GPU 3 of this host, with GPUs 2 and
# 3 agreeing to 0.1%, at a time when both were confirmed idle.
FREE_CARD_GEMM_TFLOPS = float(os.environ.get("KB_FREE_CARD_TFLOPS", "427.1"))
# Idle draw of a genuinely free card on this host, from the same session: GPUs
# 0/1/2 sit at 88-90 W at 37 C even while holding large allocations.
IDLE_POWER_W = float(os.environ.get("KB_IDLE_POWER_W", "95"))


def calibrate_card(nvml: _NVML, phase: str = "before") -> dict:
    """Measure this card's CURRENT share against a known free-card figure.

    This exists because the in-container process list cannot see a neighbour.
    NVML's ``nvmlDeviceGetComputeRunningProcesses`` returns only PIDs visible in
    the caller's PID namespace, so a container reports ``1`` -- itself -- while
    another tenant is using half the GPU. That is a second broken instrument on
    this host, alongside the utilisation counter of E22, and it fails the same
    way: it reads as "nothing else is running", which is the answer that lets a
    contaminated number through.

    What does work is arithmetic. A bf16 8192^3 GEMM is the same calibration
    E22 used, it takes about a second, and it is insensitive to everything
    except how much of the card we are actually getting. Dividing by the
    free-card figure gives ``card_share``: 1.0 means we have the card, 0.5
    means we have half of it and every absolute number in the run is worth
    about half of what it would be on a free card.

    Power is the corroborating signal. A free card on this host idles at
    ~90 W; GPU 3 measured 359 W at 55 C with no process of ours running.
    """
    snap = nvml.snapshot()
    idle_power = snap.get("power_w")
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=DEV)
    # The anchor must itself be ramped, or it reports a cold card as a busy one.
    # Measured: on a card idling at 88 W, 20 warmup iterations of this GEMM
    # (~30 ms) is not enough to leave the low clock state and the anchor read
    # 203 TFLOP/s -- half of what the same card delivered seconds later. That
    # would have been reported as "card_share 0.47, another tenant is here".
    # So: keep pushing work until the peak stops improving.
    def _round() -> float:
        vals = []
        for _ in range(6):
            ms = _event_time_ms(lambda: a.__matmul__(b), 5)
            vals.append(2.0 * 8192 ** 3 / (ms * 1e-3) / 1e12)
        return max(vals)

    ramp = []
    try:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()
        peak = _round()
        ramp.append(round(peak, 1))
        for _ in range(9):
            nxt = _round()
            ramp.append(round(nxt, 1))
            improved = (nxt - peak) / peak if peak else 0.0
            peak = max(peak, nxt)
            if improved <= 0.02:
                break
        median = peak
    finally:
        del a, b
        torch.cuda.empty_cache()

    share = peak / FREE_CARD_GEMM_TFLOPS
    # Power only means anything BEFORE we have run anything: afterwards the card
    # is hot from our own work, and reading that as a co-tenant would flag every
    # sweep. The `after` anchor relies on card_share alone.
    busy = (phase == "before" and idle_power is not None
            and idle_power > IDLE_POWER_W)
    return {
        "anchor": "bf16 8192^3 GEMM",
        "phase": phase,
        "free_card_reference_tflop_s": FREE_CARD_GEMM_TFLOPS,
        "free_card_reference_source": "kda-sm120/docs/log.md E22 (card confirmed idle)",
        "measured_peak_tflop_s": round(peak, 1),
        "measured_median_tflop_s": round(median, 1),
        "ramp_tflop_s_per_round": ramp,
        "ramped": len(ramp) < 10,
        "card_share": round(share, 3),
        "power_w": idle_power,
        "power_is_evidence": phase == "before",
        "idle_power_reference_w": IDLE_POWER_W,
        "power_suggests_other_tenant": busy,
        "process_list_is_blind": (
            "NVML compute-process count inside a container only sees this "
            "container's PIDs; it cannot detect a co-tenant. Do not use it."
        ),
        "verdict": (
            "card effectively free; absolute numbers comparable to a free-card "
            "baseline" if share >= 0.85 and not busy else
            f"CARD SHARED: we are getting ~{share:.0%} of it. Absolute TFLOP/s "
            "and GB/s in this run are depressed by roughly that factor. Compare "
            "kernels WITHIN this run; do not compare across runs or against a "
            "free-card figure without renormalising."
        ),
    }


def gpu_identity(nvml: _NVML) -> dict:
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return {
        "name": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "l2_cache_mib": round(l2_bytes() / 2**20, 1),
        "total_mem_gib": round(total / 2**30, 2),
        "free_mem_gib_at_start": round(free / 2**30, 2),
        "shared_with_other_tenants": True,
        "shared_note": (
            "GPU 3 on this host is shared with other agents; timing spread and "
            "cross-run absolute comparability are affected. Compare within a run."
        ),
        **nvml.snapshot(),
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class Timing:
    samples_ms: list[float]
    inner: int
    warmup: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    spread: float          # (p90 - p10) / median
    stationary: bool
    stationary_detail: str
    stationary_values: dict
    retried: bool

    def to_json(self) -> dict:
        return {
            "median_ms": self.median_ms,
            "p10_ms": self.p10_ms,
            "p90_ms": self.p90_ms,
            "min_ms": self.min_ms,
            "spread_p10_p90": self.spread,
            "inner_iters_per_sample": self.inner,
            "warmup_iters": self.warmup,
            "n_samples": len(self.samples_ms),
            "stationary": self.stationary,
            "stationary_detail": self.stationary_detail,
            "stationary_values": self.stationary_values,
            "retried_with_longer_warmup": self.retried,
            "samples_ms": [round(x, 6) for x in self.samples_ms],
        }


def _event_time_ms(fn: Callable[[], Any], inner: int) -> float:
    """Mean ms per call over ``inner`` back-to-back calls, by CUDA events.

    Wall clock would include Python dispatch and, worse, would be measuring the
    launch queue rather than the kernel once the queue runs ahead. Events are
    recorded on the same stream as the work, so the interval is device time.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(inner):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / inner


def _check_stationary(series: list[float]) -> tuple[bool, str, dict]:
    if _HAVE_GATES:
        g = gate_stationary(series, drift_tol=DRIFT_TOL,
                            envelope_tol=ENVELOPE_TOL, swing_floor=SWING_FLOOR)
        return g.ok, g.detail, g.values
    # Degraded standalone fallback: drift only, and say so.
    if len(series) < 6:
        return False, f"only {len(series)} samples; need >= 6", {}
    half = len(series) // 2
    a, b = st.mean(series[:half]), st.mean(series[half:])
    drift = (b - a) / a if a else 0.0
    ok = abs(drift) <= DRIFT_TOL
    return ok, (f"drift {drift:+.1%} (kda.gates unavailable; drift-only check)"
                if not ok else "settled (drift-only check)"), {"drift": drift}


# Inner-loop sizes are snapped to this coarse ladder. See _choose_inner.
_INNER_LADDER = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200)


def _choose_inner(fn: Callable[[], Any]) -> int:
    """Pick the inner-loop length deterministically.

    This is not cosmetic. The inner loop length CHANGES THE ANSWER for any
    kernel whose working set straddles L2: with ``inner=1`` every timed call
    starts with a cold cache, with ``inner=2`` the second call is partly L2-fed,
    and the per-call mean moves accordingly. An earlier version of this function
    sized the loop from a single 3-iteration probe, and the probe noise was
    enough to flip ``quantize_fp8`` on a 4096x17408 tensor between inner=1
    (5.09 ms/call) and inner=2 (2.79 ms/call) -- an 1.8x swing across repeats of
    an identical case, with the SM clock pinned at 2422 MHz and no other process
    on the card. Both numbers were stationary. Neither was reproducible.

    So: probe with a median of 5 (not a mean of 3), and snap the result to a
    coarse ladder, so ordinary probe noise cannot move the choice. The chosen
    value is reported and compared across repeats.
    """
    probes = sorted(_event_time_ms(fn, 1) for _ in range(5))
    probe = probes[len(probes) // 2]
    want = TARGET_REP_MS / max(probe, 1e-6)
    for step in _INNER_LADDER:
        if step >= want:
            return min(step, MAX_INNER)
    return MAX_INNER


def time_kernel(fn: Callable[[], Any], *, samples: int = SAMPLES,
                warmup: int = WARMUP_ITERS) -> Timing:
    """Warm up, size the inner loop, take ``samples`` CUDA-event samples.

    Re-runs once with a 3x longer warmup if the series is not stationary; the
    overwhelmingly common cause is the SM clock still ramping on a card that
    was idle, and a longer warmup fixes it. If the second attempt is still
    moving, the caller must not report the number.
    """
    retried = False
    for attempt in (0, 1):
        w = warmup * (3 if attempt else 1)
        s = samples + (samples // 2 if attempt else 0)
        for _ in range(w):
            fn()
        torch.cuda.synchronize()

        inner = _choose_inner(fn)
        series = [_event_time_ms(fn, inner) for _ in range(s)]
        ok, detail, values = _check_stationary(series)
        if ok or attempt == 1:
            ordered = sorted(series)
            n = len(ordered)
            p10 = ordered[max(0, int(0.10 * (n - 1)))]
            p90 = ordered[min(n - 1, int(math.ceil(0.90 * (n - 1))))]
            med = st.median(series)
            return Timing(
                samples_ms=series, inner=inner, warmup=w,
                median_ms=med, p10_ms=p10, p90_ms=p90, min_ms=ordered[0],
                spread=(p90 - p10) / med if med else float("nan"),
                stationary=ok, stationary_detail=detail,
                stationary_values=values, retried=retried,
            )
        retried = True
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def _to_compare(x: Any) -> torch.Tensor:
    """Reduce a kernel result to the one tensor being compared.

    Attention kernels may return ``(out, lse)``. Only ``out`` is compared, and
    that choice is recorded in the case, not inferred silently per candidate --
    two candidates compared on different members of the tuple would produce a
    correctness verdict about nothing.
    """
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor result, got {type(x).__name__}")
    if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return x.float()
    if x.dtype == torch.uint8 or x.dtype == torch.int8:
        return x.float()
    return x.float()


def compare(cand: Any, ref: Any) -> dict:
    """max abs diff, mean abs diff, relative L2, cosine similarity.

    Read ``rel_l2`` first. Cosine similarity is invariant to a global scale
    factor, so a kernel that applies the wrong ``softmax_scale`` or drops an
    ``alpha`` scores 1.000000 on cosine and is still wrong. ``max_abs_diff`` is
    also uninterpretable on its own, so the reference's own magnitude
    (``ref_absmax``) is reported next to it.
    """
    a = _to_compare(cand).flatten()
    b = _to_compare(ref).flatten()
    if a.shape != b.shape:
        return {"shape_mismatch": [list(_to_compare(cand).shape),
                                   list(_to_compare(ref).shape)]}
    d = (a - b).abs()
    aa = torch.sum(a * a, dtype=torch.float64)
    bb = torch.sum(b * b, dtype=torch.float64)
    ab = torch.sum(a * b, dtype=torch.float64)
    denom = float(torch.sqrt(aa) * torch.sqrt(bb))
    l2d = float(torch.sqrt(torch.sum(d * d, dtype=torch.float64)))
    l2b = float(torch.sqrt(bb))
    ref_absmax = float(b.abs().max())
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(torch.mean(d.double())),
        "rel_l2": (l2d / l2b) if l2b else float("nan"),
        "cosine": (float(ab) / denom) if denom else float("nan"),
        "ref_absmax": ref_absmax,
        "max_abs_diff_over_ref_absmax": (float(d.max()) / ref_absmax
                                         if ref_absmax else float("nan")),
        "nonfinite_in_candidate": int((~torch.isfinite(a)).sum()),
    }


def _checksum(t: torch.Tensor) -> float | None:
    """Byte-level checksum, so an in-place write to an input is detectable even
    when it changes values the op would not otherwise read."""
    try:
        return float(t.contiguous().view(torch.uint8).sum(dtype=torch.float64))
    except Exception:
        try:
            return float(t.double().sum())
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One operator at one shape.

    ``reference`` is mandatory and is a (name, callable) pair. The name is
    written into every numerics row. ``harness:*`` means an fp32 torch
    implementation defined in this file; ``registry:*`` means a kernel from the
    registry designated as ground truth.
    """

    name: str
    family: str
    mode: str
    model: str
    shape: dict
    unit: str                       # "TFLOP/s" or "GB/s" -- the primary one
    unit_reason: str
    build: Callable[[], dict]       # -> {"inputs": {...}, "flops": f, "bytes": b,
                                    #     "call": fn(**dispatch_overrides)}
    reference: tuple[str, Callable[[dict], Any]]
    notes: str = ""


# --- reference implementations ---------------------------------------------
# Independent of the kernels under test on purpose. For attention on sm_120
# there is only ONE registered candidate for prefill, so a candidate-vs-
# candidate comparison would be vacuous; these compute the operator's
# definition in fp32 from the same inputs.


def _gather_paged(cache: torch.Tensor, page_table: torch.Tensor,
                  b: int, length: int) -> torch.Tensor:
    """[num_pages, page_size, H, D] + page table -> [length, H, D] for seq b."""
    page_size = cache.shape[1]
    npages = (length + page_size - 1) // page_size
    pages = page_table[b, :npages].long()
    return cache[pages].reshape(-1, cache.shape[2], cache.shape[3])[:length]


def _sdpa_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               scale: float, causal_offset: int | None) -> torch.Tensor:
    """q [Lq,Hq,D], k/v [Lk,Hkv,D] -> [Lq,Hq,D], GQA-expanded, fp32.

    ``causal_offset`` is ``Lk - Lq`` for a causal suffix, or None for full
    attention. Masking is written out rather than delegated so that the FLOP
    count below can be derived from the same rule.
    """
    Lq, Hq, D = q.shape
    Lk, Hkv, _ = k.shape
    g = Hq // Hkv
    kf = k.float().repeat_interleave(g, dim=1)
    vf = v.float().repeat_interleave(g, dim=1)
    qf = q.float()
    s = torch.einsum("qhd,khd->hqk", qf, kf) * scale
    if causal_offset is not None:
        qi = torch.arange(Lq, device=q.device).view(-1, 1)
        ki = torch.arange(Lk, device=q.device).view(1, -1)
        s = s.masked_fill((ki > qi + causal_offset).unsqueeze(0), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, vf)


def _causal_pairs(lengths: list[int]) -> int:
    return sum(L * (L + 1) // 2 for L in lengths)


def _attn_flops(pairs: int, hq: int, d: int) -> float:
    """2 GEMMs (QK^T and PV), 2 FLOP per MAC, over attended (q,k) pairs."""
    return 2.0 * 2.0 * pairs * hq * d


# ---------------------------------------------------------------------------
# Qwen3.5-27B geometry
# ---------------------------------------------------------------------------

MODEL = "Qwen3.5-27B"
HQ, HKV, HD = 24, 4, 256
PAGE_SIZE = 832
HIDDEN, INTERMEDIATE = 5120, 17408
DT = torch.bfloat16
ESZ = 2  # bf16 bytes


def _cu(lengths: list[int]) -> tuple[torch.Tensor, list[int]]:
    cu = [0]
    for L in lengths:
        cu.append(cu[-1] + L)
    return torch.tensor(cu, dtype=torch.int32, device=DEV), cu


def build_mha_prefill(batch: int, seqlen: int) -> Callable[[], dict]:
    def build() -> dict:
        import tokenspeed_kernel.ops.attention as A

        lengths = [seqlen] * batch
        cu, cu_cpu = _cu(lengths)
        total = sum(lengths)
        q = torch.randn(total, HQ, HD, dtype=DT, device=DEV)
        k = torch.randn(total, HKV, HD, dtype=DT, device=DEV)
        v = torch.randn(total, HKV, HD, dtype=DT, device=DEV)
        scale = 1.0 / math.sqrt(HD)
        inputs = {"q": q, "k": k, "v": v}

        def call(**disp):
            return A.mha_prefill(q=q, k=k, v=v, cu_seqlens=cu,
                                 cu_seqlens_cpu=cu_cpu, max_seqlen=seqlen, **disp)

        def ref(_):
            outs = []
            for b in range(batch):
                s, e = cu_cpu[b], cu_cpu[b + 1]
                outs.append(_sdpa_fp32(q[s:e], k[s:e], v[s:e], scale, 0))
            return torch.cat(outs, 0)

        pairs = _causal_pairs(lengths)
        rd = total * HQ * HD * ESZ + 2 * total * HKV * HD * ESZ
        wr = total * HQ * HD * ESZ
        return {"inputs": inputs, "call": call, "ref": ref,
                "flops": _attn_flops(pairs, HQ, HD), "bytes": rd + wr,
                "attended_pairs": pairs}

    return build


def build_mha_extend(batch: int, qlen: int, ctx: int) -> Callable[[], dict]:
    def build() -> dict:
        import tokenspeed_kernel.ops.attention as A

        kv_len = ctx + qlen
        ppseq = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch * ppseq
        total_q = batch * qlen
        q = torch.randn(total_q, HQ, HD, dtype=DT, device=DEV)
        kc = torch.randn(num_pages, PAGE_SIZE, HKV, HD, dtype=DT, device=DEV)
        vc = torch.randn(num_pages, PAGE_SIZE, HKV, HD, dtype=DT, device=DEV)
        page_table = torch.arange(num_pages, dtype=torch.int32,
                                  device=DEV).view(batch, ppseq)
        cache_seqlens = torch.full((batch,), kv_len, dtype=torch.int32, device=DEV)
        cu_q, _ = _cu([qlen] * batch)
        cu_kv, _ = _cu([kv_len] * batch)
        scale = 1.0 / math.sqrt(HD)
        inputs = {"q": q, "k_cache": kc, "v_cache": vc,
                  "page_table": page_table, "cache_seqlens": cache_seqlens}

        def call(**disp):
            return A.mha_extend_with_kvcache(
                q=q, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv,
                k_cache=kc, v_cache=vc, page_table=page_table,
                cache_seqlens=cache_seqlens, max_seqlen_q=qlen,
                max_seqlen_k=kv_len, is_causal=True, **disp)

        def ref(_):
            outs = []
            for b in range(batch):
                kb = _gather_paged(kc, page_table, b, kv_len)
                vb = _gather_paged(vc, page_table, b, kv_len)
                qb = q[b * qlen:(b + 1) * qlen]
                outs.append(_sdpa_fp32(qb, kb, vb, scale, kv_len - qlen))
            return torch.cat(outs, 0)

        # causal suffix: query i attends kv 0..(kv_len-qlen+i)
        pairs = batch * (qlen * (kv_len - qlen) + qlen * (qlen + 1) // 2)
        kv_read = 2 * batch * kv_len * HKV * HD * ESZ
        return {"inputs": inputs, "call": call, "ref": ref,
                "flops": _attn_flops(pairs, HQ, HD),
                "bytes": kv_read + total_q * HQ * HD * ESZ * 2,
                "attended_pairs": pairs}

    return build


def build_mha_decode(batch: int, ctx: int, qlen: int = 1) -> Callable[[], dict]:
    def build() -> dict:
        import tokenspeed_kernel.ops.attention as A

        ppseq = (ctx + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch * ppseq
        total_q = batch * qlen
        q = torch.randn(total_q, HQ, HD, dtype=DT, device=DEV)
        kc = torch.randn(num_pages, PAGE_SIZE, HKV, HD, dtype=DT, device=DEV)
        vc = torch.randn(num_pages, PAGE_SIZE, HKV, HD, dtype=DT, device=DEV)
        page_table = torch.arange(num_pages, dtype=torch.int32,
                                  device=DEV).view(batch, ppseq)
        cache_seqlens = torch.full((batch,), ctx, dtype=torch.int32, device=DEV)
        scale = 1.0 / math.sqrt(HD)
        inputs = {"q": q, "k_cache": kc, "v_cache": vc,
                  "page_table": page_table, "cache_seqlens": cache_seqlens}

        def call(**disp):
            return A.mha_decode_with_kvcache(
                q=q, k_cache=kc, v_cache=vc, page_table=page_table,
                cache_seqlens=cache_seqlens, max_seqlen_k=ctx,
                max_seqlen_q=qlen, **disp)

        def ref(_):
            outs = []
            for b in range(batch):
                kb = _gather_paged(kc, page_table, b, ctx)
                vb = _gather_paged(vc, page_table, b, ctx)
                qb = q[b * qlen:(b + 1) * qlen]
                outs.append(_sdpa_fp32(qb, kb, vb, scale,
                                       ctx - qlen if qlen > 1 else None))
            return torch.cat(outs, 0)

        pairs = (batch * qlen * ctx if qlen == 1 else
                 batch * (qlen * (ctx - qlen) + qlen * (qlen + 1) // 2))
        kv_read = 2 * batch * ctx * HKV * HD * ESZ
        return {"inputs": inputs, "call": call, "ref": ref,
                "flops": _attn_flops(pairs, HQ, HD),
                "bytes": kv_read + total_q * HQ * HD * ESZ * 2,
                "attended_pairs": pairs}

    return build


def _ref_mm_fp32(A_: torch.Tensor, B_nk: torch.Tensor,
                 out_dtype: torch.dtype) -> torch.Tensor:
    """fp32 reference for A[M,K] @ B[N,K]^T, chunked over N.

    Chunked because the unchunked form materialises an fp32 copy of a
    [34816, 5120] weight, and this card is shared -- a harness that OOMs a
    neighbour is worse than a slow one.
    """
    M, K = A_.shape
    N = B_nk.shape[0]
    af = A_.float()
    out = torch.empty(M, N, dtype=torch.float32, device=A_.device)
    step = max(1, min(N, (1 << 26) // max(K, 1)))
    for s in range(0, N, step):
        e = min(N, s + step)
        out[:, s:e] = af @ B_nk[s:e].float().T
    return out.to(out_dtype)


def build_mm(M: int, N: int, K: int, label: str) -> Callable[[], dict]:
    def build() -> dict:
        import tokenspeed_kernel.ops.gemm as G

        a = torch.randn(M, K, dtype=DT, device=DEV) / math.sqrt(K)
        b = torch.randn(N, K, dtype=DT, device=DEV) / math.sqrt(K)
        inputs = {"A": a, "B": b}

        def call(**disp):
            return G.mm(a, b, **disp)

        def ref(_):
            return _ref_mm_fp32(a, b, DT)

        return {"inputs": inputs, "call": call, "ref": ref,
                "flops": 2.0 * M * N * K,
                "bytes": (M * K + N * K + M * N) * ESZ}

    return build


def build_mm_fp8_blockscale(M: int, N: int, K: int) -> Callable[[], dict]:
    """fp8 128x128 block-scaled mm.

    Included because it is the only operator in this sweep with more than two
    admissible kernels, so it is the one that actually exercises candidate
    ranking and cross-candidate numerics. The reference is the registered
    ``torch_mm_fp8_blockscale``, which is what the in-tree numerics suite
    treats as ground truth for this format.
    """
    def build() -> dict:
        import tokenspeed_kernel.ops.gemm as G

        f8 = torch.float8_e4m3fn
        a = (torch.randn(M, K, device=DEV) / math.sqrt(K)).to(f8)
        b = (torch.randn(N, K, device=DEV) / math.sqrt(K)).to(f8)
        a_s = torch.rand(M, K // 128, device=DEV, dtype=torch.float32) * 0.5 + 0.75
        b_s = torch.rand(N // 128, K // 128, device=DEV,
                         dtype=torch.float32) * 0.5 + 0.75
        inputs = {"A": a, "B": b, "A_scales": a_s, "B_scales": b_s}

        def call(**disp):
            return G.mm(a, b, A_scales=a_s, B_scales=b_s, out_dtype=DT,
                        quant="mxfp8", block_size=[128, 128], **disp)

        def ref(_):
            return G.mm(a, b, A_scales=a_s, B_scales=b_s, out_dtype=DT,
                        quant="mxfp8", block_size=[128, 128],
                        override="torch_mm_fp8_blockscale")

        return {"inputs": inputs, "call": call, "ref": ref,
                "flops": 2.0 * M * N * K,
                "bytes": (M * K + N * K) * 1 + M * N * ESZ}

    return build


def build_bmm(bsz: int, M: int, N: int, K: int) -> Callable[[], dict]:
    def build() -> dict:
        import tokenspeed_kernel.ops.gemm as G

        a = torch.randn(bsz, M, K, dtype=DT, device=DEV) / math.sqrt(K)
        b = torch.randn(bsz, N, K, dtype=DT, device=DEV) / math.sqrt(K)
        inputs = {"A": a, "B": b}

        def call(**disp):
            return G.bmm(a, b, **disp)

        def ref(_):
            return torch.bmm(a.float(), b.float().transpose(1, 2)).to(DT)

        return {"inputs": inputs, "call": call, "ref": ref,
                "flops": 2.0 * bsz * M * N * K,
                "bytes": (bsz * M * K + bsz * N * K + bsz * M * N) * ESZ}

    return build


def build_quant_fp8(rows: int, cols: int, with_scale: bool) -> Callable[[], dict]:
    def build() -> dict:
        import tokenspeed_kernel.ops.quantization as Q

        x = torch.randn(rows, cols, dtype=DT, device=DEV)
        scale = 0.5 if with_scale else None
        inputs = {"x": x}

        def call(**disp):
            return Q.quantize_fp8(x, scale=scale, **disp)

        def ref(_):
            y = x.float()
            if scale is not None:
                y = y / scale
            return y.to(torch.float8_e4m3fn)

        # bf16 in, fp8 out: this is a pure streaming op, so GB/s is the only
        # number that means anything. FLOPs are reported as the divide count
        # and are not a useful figure of merit.
        return {"inputs": inputs, "call": call, "ref": ref,
                "flops": float(rows * cols) if with_scale else 0.0,
                "bytes": rows * cols * (ESZ + 1)}

    return build


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

QKV_N = (HQ + 2 * HKV) * HD          # fused qkv projection out features
O_K = HQ * HD                        # o_proj in features


def build_cases() -> list[Case]:
    A_REF = ("harness:torch_sdpa_fp32",
             "fp32 einsum + explicit mask + fp32 softmax, GQA expanded by "
             "repeat_interleave; paged KV gathered through the same page table "
             "the kernel is given")
    G_REF = ("harness:torch_matmul_fp32",
             "torch.matmul on fp32-upcast operands, chunked over N, result cast "
             "back to bf16")
    cases: list[Case] = []

    def C(**kw):
        cases.append(Case(model=MODEL, **kw))

    # --- attention ---------------------------------------------------------
    C(name="mha_prefill_b4_s1024", family="attention", mode="mha_prefill",
      shape={"batch": 4, "seqlen": 1024, "num_q_heads": HQ, "num_kv_heads": HKV,
             "head_dim": HD, "dtype": "bfloat16", "causal": True},
      unit="TFLOP/s",
      unit_reason="compute-bound: O(L^2) work against O(L) traffic",
      build=build_mha_prefill(4, 1024), reference=A_REF,
      notes="Chunked-prefill-sized: 4 sequences of 1024 in one call.")

    C(name="mha_prefill_b1_s4096", family="attention", mode="mha_prefill",
      shape={"batch": 1, "seqlen": 4096, "num_q_heads": HQ, "num_kv_heads": HKV,
             "head_dim": HD, "dtype": "bfloat16", "causal": True},
      unit="TFLOP/s",
      unit_reason="compute-bound: O(L^2) work against O(L) traffic",
      build=build_mha_prefill(1, 4096), reference=A_REF,
      notes="One long sequence -- fewer, larger tiles than the b4 case.")

    C(name="mha_extend_b4_q128_ctx1024", family="attention",
      mode="mha_extend_with_kvcache",
      shape={"batch": 4, "qlen": 128, "ctx": 1024, "page_size": PAGE_SIZE,
             "num_q_heads": HQ, "num_kv_heads": HKV, "head_dim": HD,
             "dtype": "bfloat16", "is_causal": True},
      unit="TFLOP/s",
      unit_reason="q_len 128 gives enough arithmetic intensity to be compute-side",
      build=build_mha_extend(4, 128, 1024), reference=A_REF,
      notes="page_size 832 is forced by the GDN state row on this model.")

    C(name="mha_extend_b8_q32_ctx4096", family="attention",
      mode="mha_extend_with_kvcache",
      shape={"batch": 8, "qlen": 32, "ctx": 4096, "page_size": PAGE_SIZE,
             "num_q_heads": HQ, "num_kv_heads": HKV, "head_dim": HD,
             "dtype": "bfloat16", "is_causal": True},
      unit="GB/s",
      unit_reason=("q_len 32 against 4k of context: the KV read dominates, so "
                   "GB/s is the figure of merit and TFLOP/s is secondary"),
      build=build_mha_extend(8, 32, 4096), reference=A_REF,
      notes="Speculative-verify / short-extend shape.")

    C(name="mha_decode_b32_ctx4096", family="attention",
      mode="mha_decode_with_kvcache",
      shape={"batch": 32, "qlen": 1, "ctx": 4096, "page_size": PAGE_SIZE,
             "num_q_heads": HQ, "num_kv_heads": HKV, "head_dim": HD,
             "dtype": "bfloat16"},
      unit="GB/s",
      unit_reason="one query token per sequence: pure KV-cache streaming",
      build=build_mha_decode(32, 4096), reference=A_REF,
      notes="GB/s counts the KV bytes the kernel must read at least once.")

    C(name="mha_decode_b64_ctx2048", family="attention",
      mode="mha_decode_with_kvcache",
      shape={"batch": 64, "qlen": 1, "ctx": 2048, "page_size": PAGE_SIZE,
             "num_q_heads": HQ, "num_kv_heads": HKV, "head_dim": HD,
             "dtype": "bfloat16"},
      unit="GB/s", unit_reason="pure KV-cache streaming",
      build=build_mha_decode(64, 2048), reference=A_REF,
      notes="Same total KV bytes as b32_ctx4096, twice the parallelism.")

    # --- gemm --------------------------------------------------------------
    for M, tag in ((4096, "prefill"), (32, "decode")):
        C(name=f"mm_qkv_M{M}", family="gemm", mode="mm",
          shape={"M": M, "N": QKV_N, "K": HIDDEN, "dtype": "bfloat16",
                 "layer": "qkv_proj", "regime": tag},
          unit="TFLOP/s",
          unit_reason=("compute-bound at M=4096" if M >= 512 else
                       "memory-bound at M=32 (weight streaming); GB/s is the "
                       "one to read, TFLOP/s is reported for continuity"),
          build=build_mm(M, QKV_N, HIDDEN, "qkv"), reference=G_REF)

        C(name=f"mm_o_M{M}", family="gemm", mode="mm",
          shape={"M": M, "N": HIDDEN, "K": O_K, "dtype": "bfloat16",
                 "layer": "o_proj", "regime": tag},
          unit="TFLOP/s",
          unit_reason=("compute-bound at M=4096" if M >= 512 else
                       "memory-bound at M=32 (weight streaming)"),
          build=build_mm(M, HIDDEN, O_K, "o"), reference=G_REF)

        C(name=f"mm_gate_up_M{M}", family="gemm", mode="mm",
          shape={"M": M, "N": 2 * INTERMEDIATE, "K": HIDDEN, "dtype": "bfloat16",
                 "layer": "gate_up_proj", "regime": tag},
          unit="TFLOP/s",
          unit_reason=("compute-bound at M=4096" if M >= 512 else
                       "memory-bound at M=32 (weight streaming)"),
          build=build_mm(M, 2 * INTERMEDIATE, HIDDEN, "gate_up"), reference=G_REF)

        C(name=f"mm_down_M{M}", family="gemm", mode="mm",
          shape={"M": M, "N": HIDDEN, "K": INTERMEDIATE, "dtype": "bfloat16",
                 "layer": "down_proj", "regime": tag},
          unit="TFLOP/s",
          unit_reason=("compute-bound at M=4096" if M >= 512 else
                       "memory-bound at M=32 (weight streaming)"),
          build=build_mm(M, HIDDEN, INTERMEDIATE, "down"), reference=G_REF)

    C(name="mm_fp8_blockscale_M4096_qkv", family="gemm", mode="mm",
      shape={"M": 4096, "N": QKV_N, "K": HIDDEN, "quant": "mxfp8",
             "block_size": [128, 128], "layer": "qkv_proj"},
      unit="TFLOP/s", unit_reason="compute-bound",
      build=build_mm_fp8_blockscale(4096, QKV_N, HIDDEN),
      reference=("registry:torch_mm_fp8_blockscale",
                 "the in-tree reference-band kernel for the mxfp8 128x128 "
                 "block-scaled format; dequantises to fp32 and matmuls"),
      notes="The only operator in this sweep with 4 admissible kernels.")

    # --- bmm ---------------------------------------------------------------
    C(name="bmm_b24_m1024_n256_k256", family="gemm", mode="bmm",
      shape={"batch": HQ, "M": 1024, "N": HD, "K": HD, "dtype": "bfloat16"},
      unit="TFLOP/s", unit_reason="compute-bound",
      build=build_bmm(HQ, 1024, HD, HD), reference=G_REF,
      notes="Per-head batched shape, batch = num_q_heads.")

    C(name="bmm_b4_m4096_n5120_k5120", family="gemm", mode="bmm",
      shape={"batch": 4, "M": 4096, "N": HIDDEN, "K": HIDDEN, "dtype": "bfloat16"},
      unit="TFLOP/s", unit_reason="compute-bound",
      build=build_bmm(4, 4096, HIDDEN, HIDDEN), reference=G_REF,
      notes="Large batched GEMM; the shape a batched-expert path would hit.")

    # --- quantization ------------------------------------------------------
    Q_REF = ("harness:torch_fp8_cast_fp32",
             "x.float() (divided by the scale when one is given) cast to "
             "float8_e4m3fn by torch")
    C(name="quant_fp8_4096x5120", family="quantization", mode="fp8",
      shape={"rows": 4096, "cols": HIDDEN, "dtype": "bfloat16",
             "has_scale": False},
      unit="GB/s",
      unit_reason="pure streaming cast: read bf16, write fp8, no reuse",
      build=build_quant_fp8(4096, HIDDEN, False), reference=Q_REF)

    C(name="quant_fp8_scaled_4096x17408", family="quantization", mode="fp8",
      shape={"rows": 4096, "cols": INTERMEDIATE, "dtype": "bfloat16",
             "has_scale": True},
      unit="GB/s", unit_reason="pure streaming cast with a scalar divide",
      build=build_quant_fp8(4096, INTERMEDIATE, True), reference=Q_REF,
      notes="MoE/FFN-sized activation quantization.")

    return cases


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_candidates(case: Case, built: dict) -> dict:
    """Every kernel the platform admits, staged so nothing disappears silently.

    Stages, in order: registered -> capability-admitted -> signature-admitted
    -> trait-admitted. A kernel that drops out at any stage is listed with the
    stage that dropped it. The alternative -- printing only the survivors --
    is how ``sm_120 is missing from this capability table`` stays invisible for
    a month.
    """
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import spec_matches_traits

    plat = current_platform()
    reg = KernelRegistry.get()
    registered = reg._by_operator.get((case.family, case.mode), [])

    sig, traits = _signature_and_traits(case, built)

    rows = []
    admitted = []
    for spec in registered:
        cap_ok = spec.capability.satisfied_by(plat)
        sig_ok = spec.supports_format_signature(sig) if sig is not None else None
        trait_ok = spec_matches_traits(spec, traits) if traits else True
        if not cap_ok:
            stage = "capability"
        elif sig_ok is False:
            stage = "format_signature"
        elif not trait_ok:
            stage = "traits"
        else:
            stage = None
            admitted.append(spec.name)
        rows.append({
            "kernel": spec.name, "solution": spec.solution,
            "priority": spec.priority,
            "features": sorted(spec.features),
            "capability_ok": cap_ok,
            "format_signature_ok": sig_ok,
            "traits_ok": trait_ok,
            "rejected_at": stage,
            "capability": str(spec.capability),
        })
    return {
        "platform": {"device_name": plat.device_name, "arch": plat.arch,
                     "vendor": plat.vendor,
                     "features": sorted(getattr(plat, "features", []) or [])},
        "format_signature": str(sig) if sig is not None else None,
        "traits_requested": {k: str(v) for k, v in (traits or {}).items()},
        "n_registered": len(registered),
        "n_admitted": len(admitted),
        "admitted": admitted,
        "all": rows,
    }


def _signature_and_traits(case: Case, built: dict) -> tuple[Any, dict]:
    """Reproduce the dispatcher's own signature/trait derivation.

    Deliberately mirrors the ``select_kernel`` call inside each public op
    rather than guessing, so that ``admitted`` here is the same set the
    dispatcher would consider. If these ever diverge, the enumeration is
    wrong and the sweep silently under-reports candidates -- which is why the
    runner also records, per candidate, whether ``override`` was needed to
    reach it.
    """
    ins = built["inputs"]
    fam, mode = case.family, case.mode
    if fam == "attention":
        from tokenspeed_kernel.ops.attention import _attention_format_signature

        if mode == "mha_prefill":
            sig = _attention_format_signature(q=ins["q"], k=ins["k"], v=ins["v"])
            traits = {"head_dim": ins["q"].shape[-1], "sliding_window": False,
                      "support_logit_cap": False, "support_sinks": False,
                      "return_lse": False}
        else:
            sig = _attention_format_signature(
                q=ins["q"], k_cache=ins["k_cache"], v_cache=ins["v_cache"])
            traits = {"head_dim": ins["q"].shape[-1],
                      "page_size": ins["k_cache"].shape[1],
                      "sliding_window": False, "support_logit_cap": False,
                      "support_sinks": False, "return_lse": False}
            if mode == "mha_extend_with_kvcache":
                traits["is_causal"] = True
        return sig, traits
    if fam == "gemm":
        from tokenspeed_kernel.ops.gemm import _gemm_format_signature

        A_, B_ = ins["A"], ins["B"]
        quant = case.shape.get("quant")
        bs = case.shape.get("block_size")
        sig = _gemm_format_signature(A_, B_, ins.get("A_scales"),
                                     ins.get("B_scales"), DT, quant, bs)
        if mode == "bmm":
            N, K = B_.shape[1], B_.shape[2]
        else:
            N, K = B_.shape[0], A_.shape[-1]
        return sig, {
            "n_align_16": N % 16 == 0, "k_align_16": K % 16 == 0,
            "k_align_32": K % 32 == 0, "n_align_64": N % 64 == 0,
            "n_align_128": N % 128 == 0, "k_align_64": K % 64 == 0,
            "k_align_128": K % 128 == 0, "n_min_128": N >= 128,
            "k_min_128": K >= 128,
        }
    if fam == "quantization":
        from tokenspeed_kernel.signature import dense_tensor_format, format_signature

        return (format_signature(x=dense_tensor_format(ins["x"].dtype)),
                {"has_scale": bool(case.shape.get("has_scale"))})
    return None, {}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


# Measured on this host, both free cards agreeing to 0.1%, by a 1 GiB
# device-to-device clone (log.md E22). Used only as a plausibility ceiling for
# reported GB/s -- a number above it is evidence the byte accounting or the
# cache assumption is wrong, never evidence of a fast kernel.
DEVICE_CLONE_GBPS = float(os.environ.get("KB_DEVICE_GBPS", "1451"))

HIGH_SPREAD = float(os.environ.get("KB_HIGH_SPREAD", "0.10"))

# Max acceptable max/min ratio of a kernel's median latency across independent
# repeats of the whole case. See ``aggregate_repeats``.
REPRO_TOL = float(os.environ.get("KB_REPRO_TOL", "1.10"))


def _warnings_for(t: Timing, gbps: float | None, built: dict) -> list[str]:
    """Flags that do not invalidate the measurement but change how to read it.

    ``working set fits in L2`` is the one that matters most here and is the
    classic kernel-microbenchmark lie: calling the same GEMM back to back keeps
    its weight matrix resident in L2, so the loop measures an L2-fed kernel
    while the real forward pass streams that weight from HBM once per layer.
    It shows up as a GB/s figure above the card's measured DRAM bandwidth --
    which is exactly how it was caught here (a 32x5120x8192 mm reading
    2516 GB/s on a card that clones at 1451 GB/s).

    This is FLAGGED, not corrected: flushing L2 between iterations would change
    what is being measured in a different direction (every call cold, which the
    real model also is not). Read a flagged row as an upper bound.
    """
    w = []
    ws = built["bytes"]
    l2 = l2_bytes()
    if l2 and ws <= l2:
        w.append(f"working_set_{ws / 2**20:.0f}MiB_fits_in_L2_{l2 / 2**20:.0f}MiB"
                 ": repeated-call timing is L2-fed and optimistic vs a real "
                 "forward pass")
    if gbps and gbps > DEVICE_CLONE_GBPS:
        w.append(f"gb_s_{gbps:.0f}_exceeds_measured_device_bandwidth_"
                 f"{DEVICE_CLONE_GBPS:.0f}: traffic is being served from cache, "
                 "not DRAM")
    if t.spread == t.spread and t.spread > HIGH_SPREAD:
        w.append(f"p10_p90_spread_{t.spread:.1%}: shared card or a kernel too "
                 "short to time cleanly; ordering close to this margin is noise")
    return w


def run_case(case: Case, nvml: _NVML, *, samples: int, warmup: int) -> dict:
    torch.manual_seed(SEED)
    torch.cuda.empty_cache()
    built = case.build()
    enum = enumerate_candidates(case, built)

    ref_name, ref_desc = case.reference
    result: dict[str, Any] = {
        "case": case.name, "family": case.family, "mode": case.mode,
        "model": case.model, "shape": case.shape,
        "primary_unit": case.unit, "primary_unit_reason": case.unit_reason,
        "notes": case.notes,
        "flops": built["flops"], "bytes": built["bytes"],
        "attended_pairs": built.get("attended_pairs"),
        "reference": {"name": ref_name, "description": ref_desc},
        "enumeration": enum,
        "candidates": [],
        "gpu_before": nvml.snapshot(),
    }

    # Reference first, once, so every candidate is compared against the same
    # tensor -- recomputing it per candidate would let a drifting allocator or
    # a mutated input change the yardstick mid-table.
    try:
        ref_out = _to_compare(built["ref"](built)).clone()
        result["reference"]["ok"] = True
        result["reference"]["absmax"] = float(ref_out.abs().max())
    except Exception as exc:
        result["reference"]["ok"] = False
        result["reference"]["error"] = f"{type(exc).__name__}: {exc}"
        result["reference"]["traceback"] = traceback.format_exc(limit=6)
        ref_out = None

    pre = {k: _checksum(v) for k, v in built["inputs"].items()}

    for name in enum["admitted"]:
        row: dict[str, Any] = {"kernel": name}
        spec_row = next(r for r in enum["all"] if r["kernel"] == name)
        row["solution"] = spec_row["solution"]
        row["priority"] = spec_row["priority"]
        call = built["call"]
        try:
            out = call(override=name)
            torch.cuda.synchronize()
            row["works"] = True
        except Exception as exc:
            row["works"] = False
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)[:2000]
            row["traceback_tail"] = traceback.format_exc(limit=4)[-1200:]
            result["candidates"].append(row)
            torch.cuda.empty_cache()
            continue

        post = {k: _checksum(v) for k, v in built["inputs"].items()}
        mutated = [k for k in pre if pre[k] is not None and post[k] != pre[k]]
        row["inputs_mutated"] = mutated
        if mutated:
            # A kernel that writes an input has poisoned every later candidate's
            # comparison. Rebuild rather than carry on with a corrupted yardstick.
            row["inputs_mutated_action"] = "case rebuilt; later rows use fresh inputs"

        if ref_out is not None:
            try:
                row["numerics"] = compare(out, ref_out)
                row["numerics"]["reference"] = ref_name
                if ref_name == f"registry:{name}":
                    # The designated reference is itself a candidate. Its row is
                    # a self-comparison and proves nothing about correctness --
                    # say so rather than let a clean 0.0 read as a pass.
                    row["numerics"]["self_comparison"] = True
            except Exception as exc:
                row["numerics"] = {"error": f"{type(exc).__name__}: {exc}",
                                   "reference": ref_name}
        else:
            row["numerics"] = {"error": "reference failed; no comparison possible",
                               "reference": ref_name}
        del out
        torch.cuda.empty_cache()

        try:
            t = time_kernel(lambda: call(override=name),
                            samples=samples, warmup=warmup)
        except Exception as exc:
            row["timing_error"] = f"{type(exc).__name__}: {exc}"
            result["candidates"].append(row)
            continue

        row["timing"] = t.to_json()
        tflops = built["flops"] / (t.median_ms * 1e-3) / 1e12 if t.median_ms else None
        gbps = built["bytes"] / (t.median_ms * 1e-3) / 1e9 if t.median_ms else None
        row["observed"] = {"median_ms": t.median_ms, "tflop_s": tflops,
                           "gb_s": gbps}
        row["warnings"] = _warnings_for(t, gbps, built)
        if t.stationary:
            row["reported"] = {"median_ms": t.median_ms, "tflop_s": tflops,
                               "gb_s": gbps,
                               "primary": tflops if case.unit == "TFLOP/s" else gbps}
            row["rejected"] = None
        else:
            # Refuse. The numbers stay in `observed` for diagnosis; nothing
            # downstream may quote them.
            row["reported"] = {"median_ms": None, "tflop_s": None, "gb_s": None,
                               "primary": None}
            row["rejected"] = f"non-stationary timing: {t.stationary_detail}"

        if mutated:
            built = case.build()
            pre = {k: _checksum(v) for k, v in built["inputs"].items()}

        result["candidates"].append(row)

    result["gpu_after"] = nvml.snapshot()
    b, a = result["gpu_before"], result["gpu_after"]
    if b.get("available") and a.get("available"):
        result["contention"] = {
            "other_process_count_before": b.get("other_process_count"),
            "other_process_count_after": a.get("other_process_count"),
            "changed_mid_case": b.get("other_process_count")
            != a.get("other_process_count"),
            "sm_clock_mhz_before": b.get("sm_clock_mhz"),
            "sm_clock_mhz_after": a.get("sm_clock_mhz"),
        }

    if _HAVE_GATES:
        gates = [gate_attribution(case.model, None, json.dumps(case.shape))]
        # concurrency is not a kernel-level axis; the gate wants all three, so
        # record explicitly that it does not apply rather than faking a value.
        result["gates"] = [{"name": "attribution",
                            "ok": bool(case.model and case.shape),
                            "detail": "model + shape recorded; concurrency is "
                                      "not an axis at kernel level"}]
    del built
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def aggregate_repeats(reps: list[dict]) -> dict:
    """Collapse N independent repeats of one case, and gate on reproducibility.

    This gate exists because of a failure found while bringing this harness up:
    ``mha_decode_b32_ctx4096`` measured 1433 GB/s in one process and 799 GB/s in
    another, an hour apart, on the same card at the same shape -- and BOTH runs
    passed the stationarity gate with a p10-p90 spread under 3%.

    That is not a contradiction. ``gate_stationary`` asks whether a series is
    still moving. A neighbour process holding a steady share of the card, or a
    different allocator/L2 state, shifts the whole series by a constant and
    leaves it perfectly flat. Stationary and reproducible are different
    properties, and only the first one has a gate in ``kda/gates.py``.

    So: repeat the entire case -- fresh tensors, fresh warmup -- and compare
    medians across repeats. If they disagree by more than ``REPRO_TOL``, the
    case does not get a single number; it gets a range and a flag. A ranking
    decision taken across a gap smaller than this spread is not supported.
    """
    base = dict(reps[0])
    base["repeats"] = len(reps)
    per: dict[str, list] = {}
    for r in reps:
        for c in r.get("candidates", []):
            if c.get("works") and (c.get("observed") or {}).get("median_ms"):
                per.setdefault(c["kernel"], []).append(c)

    agg = []
    for kernel, rows in per.items():
        meds = [r["observed"]["median_ms"] for r in rows]
        stat = [bool((r.get("timing") or {}).get("stationary")) for r in rows]
        inners = [(r.get("timing") or {}).get("inner_iters_per_sample")
                  for r in rows]
        lo, hi = min(meds), max(meds)
        ratio = hi / lo if lo else float("inf")
        # An inner-loop length that differed between repeats makes the medians
        # incomparable regardless of how close they are -- different amounts of
        # L2 reuse were measured. Say so instead of averaging the two regimes.
        same_inner = len(set(inners)) == 1
        reproducible = ratio <= REPRO_TOL and all(stat) and same_inner
        prim = [(r["observed"]["tflop_s"] if base["primary_unit"] == "TFLOP/s"
                 else r["observed"]["gb_s"]) for r in rows]
        agg.append({
            "kernel": kernel,
            "solution": rows[0].get("solution"),
            "priority": rows[0].get("priority"),
            "n_repeats_ok": len(rows),
            "median_ms_per_repeat": [round(m, 6) for m in meds],
            "median_ms_min": lo, "median_ms_max": hi,
            "median_ms_median": st.median(meds),
            "repeat_ratio_max_over_min": ratio,
            "all_repeats_stationary": all(stat),
            "inner_iters_per_repeat": inners,
            "same_inner_across_repeats": same_inner,
            "reproducible": reproducible,
            "primary_per_repeat": [None if p is None else round(p, 2)
                                   for p in prim],
            "primary_median": (st.median([p for p in prim if p is not None])
                               if any(p is not None for p in prim) else None),
            "primary_range": ([min(p for p in prim if p is not None),
                               max(p for p in prim if p is not None)]
                              if any(p is not None for p in prim) else None),
            "reported": (
                {"median_ms": st.median(meds),
                 "primary": st.median([p for p in prim if p is not None])}
                if reproducible else
                {"median_ms": None, "primary": None,
                 "rejected": (f"not reproducible across {len(rows)} repeats: "
                              f"max/min = {ratio:.2f} (tol {REPRO_TOL:.2f})"
                              if ratio > REPRO_TOL else
                              (f"inner-loop length differed across repeats "
                               f"({inners}); different L2 reuse was measured"
                               if not same_inner else
                               "a repeat had a non-stationary timing series"))}
            ),
            "numerics": rows[0].get("numerics"),
            "warnings": sorted({w for r in rows for w in (r.get("warnings") or [])}),
        })
    base["aggregate"] = agg
    base["all_repeats"] = reps
    return base


def print_aggregate(res: dict) -> None:
    if "aggregate" not in res:
        return
    print(f"\n  --- across {res['repeats']} independent repeats "
          f"(fresh tensors + warmup each) ---")
    print(f"  {'kernel':<40}{'med ms (min..max)':>28}{'ratio':>8}"
          f"{'primary ' + res['primary_unit']:>22}  verdict")
    for a in res["aggregate"]:
        rng = f"{a['median_ms_min']:.4f}..{a['median_ms_max']:.4f}"
        pr = a["primary_range"]
        prs = f"{pr[0]:.1f}..{pr[1]:.1f}" if pr else "n/a"
        verdict = ("reproducible" if a["reproducible"]
                   else "NOT REPRODUCIBLE -- " + a["reported"]["rejected"])
        print(f"  {a['kernel']:<40}{rng:>28}{a['repeat_ratio_max_over_min']:>8.2f}"
              f"{prs:>22}  {verdict}")


def _fmt(x, spec="8.2f"):
    return "n/a".rjust(int(spec.split(".")[0])) if x is None else format(x, spec)


def print_table(res: dict) -> None:
    print()
    print("=" * 108)
    print(f"{res['case']}  [{res['family']}.{res['mode']}]  {res['model']}")
    print(f"  shape      : {json.dumps(res['shape'])}")
    print(f"  primary    : {res['primary_unit']}  ({res['primary_unit_reason']})")
    r = res["reference"]
    print(f"  REFERENCE  : {r['name']}"
          + ("" if r.get("ok") else f"   *** FAILED: {r.get('error')} ***"))
    print(f"               {r['description']}")
    e = res["enumeration"]
    print(f"  candidates : {e['n_admitted']} admitted of {e['n_registered']} "
          f"registered for this operator")
    for row in e["all"]:
        if row["rejected_at"]:
            print(f"     - {row['kernel']:<44} NOT ADMITTED "
                  f"(rejected at {row['rejected_at']})")
    print("-" * 108)
    print(f"  {'kernel':<40}{'pri':>4}{'med ms':>10}{'spread':>8}"
          f"{'TFLOP/s':>10}{'GB/s':>10}{'rel_l2':>11}{'cos':>10}  status")
    for c in res["candidates"]:
        if not c.get("works", False):
            print(f"  {c['kernel']:<40}{c['priority']:>4}"
                  f"{'':>10}{'':>8}{'':>10}{'':>10}{'':>11}{'':>10}  "
                  f"RAISES {c.get('error_type')}: {str(c.get('error'))[:60]}")
            continue
        t = c.get("timing") or {}
        n = c.get("numerics") or {}
        rep = c.get("reported") or {}
        obs = c.get("observed") or {}
        med = rep.get("median_ms")
        star = ""
        if med is None:
            med = obs.get("median_ms")
            star = "*"
        status = "ok"
        if c.get("rejected"):
            status = "REJECTED " + c["rejected"][:52]
        elif c.get("timing_error"):
            status = "TIMING-ERR " + c["timing_error"][:48]
        if c.get("inputs_mutated"):
            status += f" | MUTATES INPUTS {c['inputs_mutated']}"
        for wmsg in c.get("warnings") or []:
            status += " | " + wmsg.split(":")[0]
        print(f"  {c['kernel']:<40}{c['priority']:>4}"
              f"{_fmt(med, '9.4f')}{star:<1}"
              f"{_fmt(t.get('spread_p10_p90'), '7.1%')} "
              f"{_fmt(rep.get('tflop_s') if not star else obs.get('tflop_s'), '9.1f')} "
              f"{_fmt(rep.get('gb_s') if not star else obs.get('gb_s'), '9.1f')} "
              f"{_fmt(n.get('rel_l2'), '10.3e')}"
              f"{_fmt(n.get('cosine'), '10.6f')}  {status}")
    if any(c.get("rejected") for c in res["candidates"]):
        print("  * value shown was NOT stationary and is not reportable "
              "(diagnostic only)")
    seen: set[str] = set()
    for c in res["candidates"]:
        for wmsg in c.get("warnings") or []:
            if wmsg not in seen:
                seen.add(wmsg)
                print(f"  warn: {wmsg}")
    co = res.get("contention")
    if co and co.get("changed_mid_case"):
        print(f"  !! other compute processes changed mid-case: "
              f"{co['other_process_count_before']} -> "
              f"{co['other_process_count_after']} (shared card)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", default="all",
                    help="comma-separated case names, or 'all'")
    ap.add_argument("--family", default=None, help="filter by operator family")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    ap.add_argument("--json", default=os.environ.get("KB_JSON", ""),
                    help="write results JSON here")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each case N times end to end and gate on whether "
                         "the medians agree; 1 disables the reproducibility gate")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    ap.add_argument("--plugin", action="append", default=[],
                    help="import an out-of-tree kernel plugin module before "
                         "enumeration (repeatable)")
    args = ap.parse_args(argv)

    cases = build_cases()
    if args.list:
        for c in cases:
            print(f"{c.name:<34} {c.family}.{c.mode:<26} {c.unit:>8}  "
                  f"{json.dumps(c.shape)}")
        return 0

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 3

    import tokenspeed_kernel  # noqa: F401
    from tokenspeed_kernel.registry import load_builtin_kernels

    load_builtin_kernels()
    import tokenspeed_kernel.ops.attention  # noqa: F401
    import tokenspeed_kernel.ops.gemm  # noqa: F401
    import tokenspeed_kernel.ops.quantization  # noqa: F401

    plugins = list(args.plugin) + [
        p for p in os.environ.get("KB_PLUGINS", "").split(",") if p.strip()
    ]
    plugin_status = []
    for mod in plugins:
        try:
            __import__(mod.strip())
            plugin_status.append({"module": mod.strip(), "imported": True})
        except Exception as exc:
            # A plugin that failed to import registers nothing, and its absence
            # would otherwise read as "the plugin kernel lost on merit".
            plugin_status.append({"module": mod.strip(), "imported": False,
                                  "error": f"{type(exc).__name__}: {exc}"})
    for p in plugin_status:
        if not p["imported"]:
            print(f"!! plugin {p['module']} FAILED to import: {p['error']}",
                  file=sys.stderr)

    nvml = _NVML()
    want = None if args.cases == "all" else set(
        s.strip() for s in args.cases.split(","))
    selected = [c for c in cases
                if (want is None or c.name in want)
                and (args.family is None or c.family == args.family)]
    if not selected:
        print(f"no cases matched {args.cases!r}", file=sys.stderr)
        return 3

    import tokenspeed_kernel as _tsk

    meta = {
        "harness": HARNESS_VERSION,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": _platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tokenspeed_kernel_path": os.path.dirname(_tsk.__file__),
        "plugins": plugin_status,
        "kda_gates_available": _HAVE_GATES,
        "config": {"samples": args.samples, "warmup": args.warmup,
                   "target_rep_ms": TARGET_REP_MS, "drift_tol": DRIFT_TOL,
                   "envelope_tol": ENVELOPE_TOL, "swing_floor": SWING_FLOOR,
                   "seed": SEED},
        "gpu": gpu_identity(nvml),
        "measurement_notes": [
            "GPU utilisation is never read: nvidia-smi reports sm=100% on idle "
            "GPUs 2 and 3 of this host (kda-sm120/docs/log.md E22).",
            "The card is shared with other agents; timing spread reflects that.",
            "Latency is CUDA-event device time over an inner loop, median of N "
            "samples; a non-stationary series is not reportable.",
            f"GB/s above {DEVICE_CLONE_GBPS:.0f} (this host's measured "
            "device-to-device clone bandwidth, log.md E22) means the traffic is "
            "coming from L2, not DRAM -- see per-candidate `warnings`.",
        ],
    }
    for k in ("flashinfer", "triton"):
        try:
            meta[k] = __import__(k).__version__
        except Exception:
            meta[k] = None

    print(json.dumps(meta["gpu"], indent=2))
    # Anchor before and after: a neighbour that arrives or leaves mid-sweep
    # invalidates comparisons between the first cases and the last ones, and
    # nothing else in the run would show it.
    meta["calibration_before"] = calibrate_card(nvml)
    print("\n=== card calibration (before) ===")
    print(json.dumps(meta["calibration_before"], indent=2))

    results = []
    for c in selected:
        try:
            reps = [run_case(c, nvml, samples=args.samples, warmup=args.warmup)
                    for _ in range(max(1, args.repeats))]
            res = aggregate_repeats(reps) if args.repeats > 1 else reps[0]
        except Exception as exc:
            res = {"case": c.name, "family": c.family, "mode": c.mode,
                   "model": c.model, "shape": c.shape,
                   "case_error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(limit=8)}
            print(f"\n!! case {c.name} FAILED to set up: {res['case_error']}")
        else:
            print_table(res)
            print_aggregate(res)
        results.append(res)

    meta["calibration_after"] = calibrate_card(nvml, phase="after")
    cb, ca = meta["calibration_before"], meta["calibration_after"]
    drift = (ca["measured_peak_tflop_s"] / cb["measured_peak_tflop_s"]
             if cb["measured_peak_tflop_s"] else float("nan"))
    meta["calibration_drift"] = {
        "after_over_before": round(drift, 3),
        "stable": abs(drift - 1.0) <= 0.10,
        "note": ("card share held steady across the sweep" if abs(drift - 1.0) <= 0.10
                 else "CARD SHARE CHANGED DURING THE SWEEP: cases measured early "
                      "and late are not comparable with each other"),
    }
    print("\n=== card calibration (after) ===")
    print(json.dumps({k: meta["calibration_after"][k] for k in
                      ("measured_peak_tflop_s", "card_share", "verdict")},
                     indent=2))
    print(json.dumps(meta["calibration_drift"], indent=2))

    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = {"meta": meta, "results": results}
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"\nwritten to {args.json}")

    n_rejected = sum(1 for r in results for c in r.get("candidates", [])
                     if c.get("rejected"))
    n_raise = sum(1 for r in results for c in r.get("candidates", [])
                  if not c.get("works", True))
    n_irrepro = sum(1 for r in results for a in r.get("aggregate", [])
                    if not a["reproducible"])
    print(f"\nsummary: {len(results)} cases, "
          f"{sum(len(r.get('candidates', [])) for r in results)} candidate runs, "
          f"{n_raise} raised, {n_rejected} rejected as non-stationary, "
          f"{n_irrepro} rejected as not reproducible across repeats")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
