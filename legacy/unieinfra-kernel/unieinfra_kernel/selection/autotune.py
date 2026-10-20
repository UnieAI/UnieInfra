# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Measurement-based kernel selection: time the candidates, keep the fastest.

Why this exists
---------------
Heuristic ranking (``oracle > objective > priority > origin``) is a **belief**
about hardware. Priority bands in particular encode "the narrowly-gated
datacenter kernel is the best one", which is precisely the assumption that fails
on sm_120 — a part that is arch-*newer* than sm_90 yet has no tcgen05, so an
open-ended ``>=`` arch gate admits kernels that then die or crawl (E30, E37).
A measurement cannot be wrong about the hardware it ran on.

E34 already showed automated selection beating human curation on a *static*
gate: routing every dense ``Linear`` through the trait gate measured +4.2% on
batch-1 Qwen3-8B where a hand-picked ``qkv_proj,o_proj`` list measured +1.6%,
the difference being ``down_proj`` — a layer the human list simply forgot. This
module is the dynamic version of the same argument.

What it does
------------
:func:`autotune_select` takes the candidate set that survived capability,
signature and trait filtering, synthesizes real inputs for the requested shape
(:mod:`unieinfra_kernel.selection.inputs`), times every candidate, and returns
the fastest. Four properties make it safe to put in an engine:

* **It never runs while serving.** Every call checks
  :func:`~unieinfra_kernel.selection.tuning.autotune_frozen` and degrades to
  heuristic ranking once startup has frozen tuning. See that module for why
  tuning during serving deadlocks under tensor parallelism.
* **It is fail-safe.** Each candidate is timed inside a guard; one that raises
  (a bad arch gate, a Triton compile failure, an unsupported layout) is dropped
  from the ranking instead of taking the engine down. E30 catalogued several
  kernels that won heuristic ranking and *then* died at runtime; measurement is
  what turns that class of bug into a non-event.
* **It is cached on disk**, keyed on ``device_name`` and not merely ``arch``,
  because sm_120 spans the RTX PRO 6000 (188 SMs) and the RTX 5090 (170 SMs) at
  different clocks and they do not have to agree on a winner.
* **It defeats the L2.** GB202's L2 is 128 MiB, larger than every weight matrix
  in an 8B decode step, so timing a loop that re-reads one weight measures L2
  bandwidth and not DRAM (it reported >200% of peak before this was fixed in
  ``benchmark/scripts/bench_decode_gemv.py``). Inputs are rotated over enough
  distinct buffers that each call starts cold, exactly as a real decode step
  does when 16 GB of weights stream past between two visits to one layer.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from unieinfra_kernel.selection.inputs import get_input_generator
from unieinfra_kernel.selection.platform import PlatformInfo
from unieinfra_kernel.selection.registry import KernelSpec
from unieinfra_kernel.selection.tuning import autotune_frozen

logger = logging.getLogger(__name__)

__all__ = [
    "AutotuneParams",
    "Measurement",
    "AutotuneCache",
    "autotune_cache",
    "measure_candidates",
    "shape_bucket",
    "time_kernel",
]

# GB202's L2 is 128 MiB. Rotate over at least this many bytes of distinct inputs
# so a repeat call cannot be served from it. Same constant, and the same reason,
# as ``benchmark/scripts/bench_decode_gemv.py``.
_L2_ROTATION_BYTES = 640 * 1024 * 1024

# Hard ceiling on the transient allocation the rotation is allowed to make.
# Tuning runs after the model is resident but (by design) before the KV cache is
# sized, so this is spare memory — but it is not unlimited.
_MAX_ROTATION_BYTES = 1024 * 1024 * 1024

_CACHE_VERSION = 2


def _env(suffix: str) -> str | None:
    """Read a selection env var under either supported package prefix."""
    for prefix in ("UNIEINFRA_KERNEL_", "TOKENSPEED_KERNEL_"):
        value = os.environ.get(prefix + suffix)
        if value:
            return value
    return None


@dataclass
class AutotuneParams:
    """Tuning knobs for the ``AUTOTUNE`` selection strategy.

    Attributes:
        warmup_iters: Untimed calls before measuring, to absorb Triton JIT,
            cuBLAS handle creation and the first-touch page faults.
        bench_iters: Timed calls per repetition.
        reps: Repetitions; the reported time is the **median** across them, so a
            single scheduling hiccup cannot decide a kernel.
        use_cuda_events: Time with CUDA events (default) rather than
            ``perf_counter`` around a synchronize.
        rotate_inputs: Rotate over several independently allocated input sets so
            each call starts with a cold L2. Disable only for ops whose working
            set is already far past cache.
        max_rotation_bytes: Ceiling on the transient allocation the rotation may
            make. ``0`` disables rotation.
        margin: Relative speedup the measured winner must show over the
            heuristic winner before the decision is changed. ``0.02`` means "the
            heuristic keeps the tie unless something is 2% faster", which stops
            run-to-run noise from flipping (and then freezing, via the disk
            cache) a decision. Set to ``0.0`` for pure fastest-wins.
        use_cache: Consult and update the on-disk measurement cache.
    """

    warmup_iters: int = 5
    bench_iters: int = 20
    reps: int = 3
    use_cuda_events: bool = True
    rotate_inputs: bool = True
    max_rotation_bytes: int = _MAX_ROTATION_BYTES
    margin: float = 0.02
    use_cache: bool = True


@dataclass
class Measurement:
    """One candidate's measured result.

    Attributes:
        name: Kernel name.
        median_us: Median latency in microseconds, or ``None`` if it failed.
        error: Short description of why the candidate was dropped, if it was.
    """

    name: str
    median_us: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the candidate produced a usable timing."""
        return self.median_us is not None and self.error is None


# ---------------------------------------------------------------------------
# Shape bucketing
# ---------------------------------------------------------------------------


def shape_bucket(shape: dict[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    """Reduce a concrete shape to the key autotune results are shared across.

    The reduction is deliberately conservative:

    * ``M`` (the token count in a decode step) is rounded **up to a power of
      two**. ``M`` is what decides whether a GEMV beats a tensor-core GEMM, and
      E33 measured the crossover between ``M = 2`` and ``M = 8``, so 1, 2, 4 and
      8 must each keep their own answer — but ``M = 5`` and ``M = 6`` need not.
    * Every other dimension is kept **exact**. ``N`` and ``K`` are fixed by the
      model's architecture, so there is only a handful of them per model and
      nothing is gained by coarsening them — while bucketing them could easily
      merge ``n_le_16384`` with a shape that fails it.
    * Non-integer entries (labels and other annotations) are dropped.

    Args:
        shape: Concrete shape dict, or ``None``.

    Returns:
        A hashable, sorted tuple usable in a cache key. Empty when ``shape`` is
        ``None`` or carries no integer dimensions.
    """
    if not shape:
        return ()
    items: list[tuple[str, Any]] = []
    for key, value in sorted(shape.items()):
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        if key == "M":
            value = 1 << max(0, (max(int(value), 1) - 1).bit_length())
        items.append((key, int(value)))
    return tuple(items)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _tensor_bytes(inputs: dict[str, Any]) -> int:
    """Total bytes of the tensor arguments in one input set."""
    import torch

    total = 0
    for value in inputs.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def _rotation_count(nbytes: int, params: AutotuneParams) -> int:
    """How many distinct input sets are needed to defeat the L2."""
    if not params.rotate_inputs or params.max_rotation_bytes <= 0 or nbytes <= 0:
        return 1
    wanted = -(-_L2_ROTATION_BYTES // nbytes)  # ceil
    affordable = max(1, params.max_rotation_bytes // nbytes)
    return max(1, min(wanted, affordable, 16))


def time_kernel(
    kernel: Callable[..., Any],
    input_sets: list[dict[str, Any]],
    params: AutotuneParams,
) -> float:
    """Time one kernel over rotated input sets and return the median microseconds.

    Warms up, then runs ``params.reps`` repetitions of ``params.bench_iters``
    calls each, cycling through ``input_sets`` so consecutive calls touch
    different buffers. The result is the median of the per-repetition means.

    Args:
        kernel: The kernel implementation, called as ``kernel(**inputs)``.
        input_sets: One or more keyword-argument dicts. Cycled round-robin.
        params: Timing knobs.

    Returns:
        Median latency of a single call, in microseconds.

    Raises:
        Anything the kernel raises. The caller is expected to guard (see
        :func:`measure_candidates`).
    """
    import torch

    n_sets = len(input_sets)

    with torch.no_grad():
        for i in range(params.warmup_iters):
            kernel(**input_sets[i % n_sets])
    torch.cuda.synchronize()

    samples: list[float] = []
    for rep in range(max(1, params.reps)):
        if params.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                for i in range(params.bench_iters):
                    kernel(**input_sets[(rep + i) % n_sets])
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1e3 / params.bench_iters)
        else:
            import time

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for i in range(params.bench_iters):
                    kernel(**input_sets[(rep + i) % n_sets])
            torch.cuda.synchronize()
            samples.append(
                (time.perf_counter() - t0) * 1e6 / params.bench_iters,
            )

    return float(statistics.median(samples))


def measure_candidates(
    candidates: list[tuple[KernelSpec, Callable[..., Any]]],
    family: str,
    mode: str,
    format_signature: Any,
    shape: dict[str, Any],
    params: AutotuneParams,
    *,
    device: str = "cuda",
    seed: int = 42,
) -> list[Measurement]:
    """Time every candidate on one concrete shape.

    Args:
        candidates: ``(spec, impl)`` pairs that survived the selection filters.
        family: Operator family.
        mode: Operator mode.
        format_signature: The request's format signature, handed to the input
            generator so it synthesizes operands of the right formats.
        shape: Concrete shape dict, e.g. ``{"M": 1, "N": 6144, "K": 4096}``.
        params: Timing knobs.
        device: Device to allocate inputs on.
        seed: RNG seed, so a re-tune on the same box repeats.

    Returns:
        One :class:`Measurement` per candidate, in the input order. A candidate
        that raised has ``median_us=None`` and a populated ``error`` — it is
        *dropped*, never fatal, which is the whole point: a kernel that wins
        ranking and then dies at runtime is exactly what measurement should
        eliminate before it ever sees a request.

    Raises:
        RuntimeError: If no input generator is registered for the operator, or
            the generator cannot synthesize this signature/shape. Callers treat
            this as "not tunable" and fall back to the heuristic.
    """
    import torch

    generator = get_input_generator(family, mode)
    if generator is None:
        raise RuntimeError(f"no input generator registered for {family}.{mode}")

    try:
        probe = generator(
            format_signature=format_signature,
            shape=shape,
            device=device,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 - "cannot synthesize" is not fatal
        raise RuntimeError(
            f"input generator failed for {family}.{mode}: {exc}"
        ) from exc

    n_rot = _rotation_count(_tensor_bytes(probe), params)
    input_sets = [probe]
    for i in range(1, n_rot):
        try:
            input_sets.append(
                generator(
                    format_signature=format_signature,
                    shape=shape,
                    device=device,
                    seed=seed + i,
                )
            )
        except Exception:  # noqa: BLE001 - rotation is an optimisation, not a gate
            break

    results: list[Measurement] = []
    try:
        for spec, impl in candidates:
            try:
                median_us = time_kernel(impl, input_sets, params)
                results.append(Measurement(spec.name, median_us=median_us))
            except Exception as exc:  # noqa: BLE001 - drop, never crash
                torch.cuda.synchronize()
                logger.warning(
                    "[unieinfra_kernel:autotune] dropping %s for %s.%s %s: %s",
                    spec.name,
                    family,
                    mode,
                    shape,
                    exc,
                )
                results.append(
                    Measurement(spec.name, error=f"{type(exc).__name__}: {exc}")
                )
    finally:
        del input_sets
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    """Filesystem-safe rendering of a device name."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in text)[:96]


@dataclass
class AutotuneCache:
    """Persistent record of autotune decisions for one machine.

    The file is keyed by ``device_name`` **and** ``arch``, and each entry is
    additionally keyed by operator, format signature, traits and shape bucket.
    ``device_name`` matters independently of ``arch``: sm_120 covers both the
    RTX PRO 6000 (188 SMs, server clocks) and the RTX 5090 (170 SMs), and a
    grid-size-sensitive kernel can win on one and lose on the other.

    Attributes:
        path: JSON file backing the cache.
        entries: ``key -> {"kernel": name, "times_us": {...}}``.
    """

    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    _loaded: bool = False
    _dirty: bool = False

    @staticmethod
    def default_path(platform: PlatformInfo) -> Path:
        """Return the cache file for a platform.

        ``UNIEINFRA_KERNEL_AUTOTUNE_CACHE`` overrides the directory; the default
        is ``$XDG_CACHE_HOME/unieinfra-kernel/autotune`` (or ``~/.cache/...``).
        """
        override = _env("AUTOTUNE_CACHE")
        if override:
            directory = Path(override).expanduser()
        else:
            base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
            directory = Path(base).expanduser() / "unieinfra-kernel" / "autotune"
        return directory / f"{_slug(platform.device_name)}-{_slug(platform.arch)}.json"

    @staticmethod
    def make_key(
        platform: PlatformInfo,
        family: str,
        mode: str,
        format_signature: Any,
        traits: dict[str, Any] | None,
        shape: dict[str, Any] | None,
    ) -> str:
        """Build the cache key required by the design.

        The key is ``(device_name, arch, family, mode, format_signature, traits,
        shape-bucket)`` rendered as a stable string.
        """
        trait_items = tuple(sorted((traits or {}).items()))
        return "|".join(
            (
                platform.device_name,
                platform.arch,
                family,
                mode,
                str(format_signature),
                repr(trait_items),
                repr(shape_bucket(shape)),
            )
        )

    def load(self) -> None:
        """Read the backing file once. A corrupt or unreadable file is ignored."""
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self.path) as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return
        except Exception:  # noqa: BLE001 - a bad cache must never be fatal
            logger.warning(
                "[unieinfra_kernel:autotune] ignoring unreadable cache %s",
                self.path,
                exc_info=True,
            )
            return
        if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
            return
        entries = data.get("entries")
        if isinstance(entries, dict):
            self.entries = entries

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached record for ``key``, or ``None``."""
        self.load()
        record = self.entries.get(key)
        return record if isinstance(record, dict) else None

    def put(self, key: str, kernel: str, times_us: dict[str, float | None]) -> None:
        """Record a decision and the timings that produced it."""
        self.load()
        self.entries[key] = {"kernel": kernel, "times_us": times_us}
        self._dirty = True

    def flush(self) -> None:
        """Write the cache back, atomically. Failures are logged, never raised."""
        if not self._dirty:
            return
        payload = {"version": _CACHE_VERSION, "entries": self.entries}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w",
                dir=str(self.path.parent),
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            )
            with handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
            os.replace(handle.name, self.path)
            self._dirty = False
        except Exception:  # noqa: BLE001 - a read-only cache dir is not fatal
            logger.warning(
                "[unieinfra_kernel:autotune] could not write cache %s",
                self.path,
                exc_info=True,
            )

    def clear(self) -> None:
        """Drop every entry, in memory and on disk."""
        self.entries = {}
        self._loaded = True
        self._dirty = True
        self.flush()


_cache_by_path: dict[Path, AutotuneCache] = {}


def autotune_cache(platform: PlatformInfo) -> AutotuneCache:
    """Return the (process-wide, memoised) disk cache for a platform."""
    path = AutotuneCache.default_path(platform)
    cache = _cache_by_path.get(path)
    if cache is None:
        cache = AutotuneCache(path=path)
        _cache_by_path[path] = cache
    return cache


def autotune_available() -> bool:
    """Whether measurement is possible in this process right now.

    Returns:
        ``False`` when tuning has been frozen for serving, or when there is no
        CUDA device to measure on.
    """
    if autotune_frozen():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - no torch/CUDA is simply "not available"
        return False
