# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Minimal dense-linear façade over the capability registry.

This is the standalone equivalent of ``tokenspeed_kernel.ops.gemm.mm`` for the
one operator a vLLM base actually needs from us today: a dense
``y = x @ W.T + b``. It deliberately reproduces that façade's **trait
vocabulary** verbatim (``m_le_8``, ``n_le_16384``, the alignment/minimum
buckets), because those trait names are the contract our registered kernels gate
on — a kernel that declares ``m_le_8`` is only ever offered a request whose
trait dict says ``m_le_8: True``.

Two things make this safe to call from an engine hot path:

* **Selection is memoised per ``(M, N, K, dtype, out_dtype)``.** The first call
  for a shape runs the full pipeline (capability filter → signature filter →
  trait filter → objective/priority/origin scoring); later calls are one dict
  lookup. ``select_kernel`` has its own cache too, but building the signature
  and trait dict is itself the expensive part, so we skip it.
* **The whole thing is wrapped in a ``torch.library`` custom op**
  (:func:`torch.ops.unieinfra.linear`). ``torch.compile`` / Inductor treat it as
  opaque, so the Python-level selection never has to be traced, never causes a
  graph break, and — because vLLM captures decode CUDA graphs — runs only at
  capture time, not on every replayed step.

Set ``UNIEINFRA_KERNEL_LOG_SELECTION=1`` to have every *new* shape log the
kernel that won, which is how selection is proven rather than assumed.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading

import torch
from unieinfra_kernel.boundary import (
    NoKernelFoundError,
    ScaleFormat,
    SelectionStrategy,
    dense_tensor_format,
    format_signature,
    select_kernel,
    tensor_format,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SCALED_KINDS",
    "dump_selection_evidence",
    "layer_is_claimable",
    "gemm_format_signature",
    "linear",
    "linear_fp8",
    "linear_fp8_op",
    "linear_nvfp4",
    "linear_nvfp4_op",
    "register_engine_fallback",
    "selection_log",
    "reset_dispatch_cache",
    "shape_traits",
]

# Kernels that accept and apply ``bias`` inside their own epilogue. Mirrors
# ``tokenspeed_kernel.ops.gemm._KERNELS_WITH_FUSED_BIAS``; anything not listed
# gets a post-GEMM add instead.
_KERNELS_WITH_FUSED_BIAS: frozenset[str] = frozenset(
    {
        "torch_mm",
        "torch_linear",
        "triton_mm_fp8_scaled",
        "unieinfra_decode_gemv_triton",
        "unieinfra_decode_gemv_fp8_triton",
        "unieinfra_decode_gemv_nvfp4_triton",
        "unieinfra_decode_batch_gemm_triton",
    }
)

# Quantized weight formats this façade can dispatch. ``dense`` is the bf16/fp16
# path handled by :func:`linear`.
SCALED_KINDS: tuple[str, ...] = ("fp8", "nvfp4", "mxfp8", "mxfp4")

_LOG_SELECTION = os.environ.get("UNIEINFRA_KERNEL_LOG_SELECTION", "0") != "0"

# Where to write the per-process selection record. Selection happens inside
# vLLM's **worker** process, whose stdout is captured and whose logging config
# only wires up the ``vllm`` logger hierarchy — so neither ``logger.info`` nor
# an in-driver ``selection_log()`` call can see it. Writing one JSON file per
# PID is the only way to *prove* which kernel ran, rather than infer it from a
# timing delta.
_SELECTION_DUMP = os.environ.get("UNIEINFRA_KERNEL_SELECTION_DUMP", "").strip()


def _install_stderr_handler() -> None:
    """Give our logger its own handler when selection logging is requested.

    vLLM configures handlers on the ``vllm`` logger only, and the root logger is
    left bare in worker processes, so an un-handled ``logger.info`` is silently
    dropped. Only runs when the user explicitly asked for the log.
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


if _LOG_SELECTION:  # pragma: no cover - opt-in diagnostics
    _install_stderr_handler()

_lock = threading.Lock()
# (M, N, K, in_dtype, out_dtype) -> (kernel_name, impl, fused_bias) or None when
# no kernel matched and the caller must fall back to F.linear.
_cache: dict[tuple, tuple[str, object, bool] | None] = {}
# Ordered record of every distinct (shape -> kernel) decision this process made.
_selection_log: list[dict] = []


def shape_traits(M: int, N: int, K: int) -> dict[str, object]:
    """Return the ``gemm.mm`` trait dict for a concrete shape.

    The names and thresholds are copied from
    ``tokenspeed_kernel.ops.gemm.mm`` so a kernel registered against either base
    sees the identical vocabulary.

    Args:
        M: Rows of the activation (tokens in this forward).
        N: Output width of the projection.
        K: Reduction width.

    Returns:
        Mapping of trait name to value, suitable for ``select_kernel(traits=)``.
    """
    return {
        "m_le_8": M <= 8,
        # Finer decode buckets. Additive: a kernel that does not declare them is
        # never filtered on them, so every pre-existing registration is
        # unaffected. They exist because the *quantized* decode GEMVs must have
        # a declarative gate that matches their profitability rule exactly —
        # unlike the dense one, whose in-kernel fallback (``F.linear``) is both
        # CUDA-graph safe and identical to the incumbent, a quantized kernel's
        # fallback is a slow dequantize that the engine would never have run.
        "m_le_4": M <= 4,
        "m_le_2": M <= 2,
        # Decode-*batch* bucket. A step decoding C sequences issues M = C rows,
        # so 16..256 is the concurrency band a served engine actually runs in.
        # Kernels tuned for it (E45) declare both ends, which is what keeps
        # CUDA-graph capture -- which walks M = 1..512 -- from ever offering
        # them a shape outside the band they were measured on (E38).
        "m_min_16": M >= 16,
        "m_le_256": M <= 256,
        "n_le_16384": N <= 16384,
        # A decode projection's N sits between one and six times the hidden
        # size. ``n_le_16384`` is too tight for it (Qwen3-8B's gate_up is
        # 24576) and no upper bound at all lets a decode-shaped kernel claim
        # ``lm_head``, whose N is the vocabulary -- a different kernel problem
        # at 25x the weight matrix, and not one E45 measured.
        "n_le_32768": N <= 32768,
        "n_align_16": N % 16 == 0,
        "k_align_16": K % 16 == 0,
        "k_align_32": K % 32 == 0,
        "n_align_64": N % 64 == 0,
        "n_align_128": N % 128 == 0,
        "k_align_64": K % 64 == 0,
        "k_align_128": K % 128 == 0,
        "n_min_128": N >= 128,
        "k_min_128": K >= 128,
    }


def _scale_storage_dtype(*scales: torch.Tensor | None) -> torch.dtype:
    """Storage dtype of the first present scale, defaulting to fp32."""
    for scale in scales:
        if scale is not None:
            return scale.dtype
    return torch.float32


def _infer_fp8_scale_granularity(
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
) -> str:
    """``tensor`` when both fp8 scales are scalars, else ``channel``."""
    if A_scales is None or B_scales is None:
        return "unknown"
    if A_scales.numel() == 1 and B_scales.numel() == 1:
        return "tensor"
    return "channel"


def gemm_format_signature(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    quant: str | None,
    block_size: list[int] | None = None,
    *,
    fp8_storage_dtype: torch.dtype = torch.float8_e4m3fn,
) -> object:
    """Build the ``gemm.mm`` format signature for one request.

    This is the standalone equivalent of
    ``tokenspeed_kernel.ops.gemm._gemm_format_signature`` and reproduces its
    branches verbatim, so a kernel registered against either base matches the
    identical signature object. The dense branch is what
    :func:`linear` uses; the scaled branches are what the quantized entry points
    below use.

    When ``A_scales`` is ``None`` on a scaled ``quant`` the activation has not
    been quantized yet. Two cases follow, and they are different signatures:

    * ``A`` is still bf16/fp16 -> a **weight-only** signature (dense ``a`` role).
      A kernel that claims it consumes the activation as-is, so no online
      quantization happens at all.
    * ``A`` is already in the quantized storage dtype -> the symmetric signature
      with the scale metadata the format implies.

    Args:
        A: Activation tensor (quantized or not).
        B: Weight tensor in its storage format.
        A_scales: Activation scales, or ``None``.
        B_scales: Weight scales, or ``None``.
        quant: ``None``/``"none"`` for dense, else one of :data:`SCALED_KINDS`.
        block_size: Block shape for ``mxfp8`` block scaling.
        fp8_storage_dtype: Platform FP8-E4M3 dtype (``float8_e4m3fnuz`` on some
            AMD parts).

    Returns:
        A ``FormatSignature`` describing the request.

    Raises:
        ValueError: If ``quant`` is unknown, or required metadata is missing.
    """
    if quant in (None, "none", "dense"):
        return format_signature(
            a=dense_tensor_format(A.dtype), b=dense_tensor_format(B.dtype)
        )

    weight_only = A.dtype in (torch.bfloat16, torch.float16) and A_scales is None

    if quant == "fp8":
        scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales, B_scales),
            granularity=(
                "tensor"
                if weight_only and B_scales is not None and B_scales.numel() == 1
                else (
                    "channel"
                    if weight_only
                    else _infer_fp8_scale_granularity(A_scales, B_scales)
                )
            ),
        )
        b_format = tensor_format("scaled-fp8", B.dtype, scale=scale)
        if weight_only:
            return format_signature(a=dense_tensor_format(A.dtype), b=b_format)
        return format_signature(
            a=tensor_format(
                "scaled-fp8",
                A.dtype if A_scales is not None else fp8_storage_dtype,
                scale=scale,
            ),
            b=b_format,
        )

    if quant == "nvfp4":
        b_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(B_scales),
            granularity="block",
            block_shape=(16,),
        )
        b_format = tensor_format("nvfp4", B.dtype, scale=b_scale)
        if weight_only:
            return format_signature(a=dense_tensor_format(A.dtype), b=b_format)
        a_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales),
            granularity="block",
            block_shape=(16,),
        )
        return format_signature(
            a=tensor_format("nvfp4", A.dtype, scale=a_scale), b=b_format
        )

    if quant == "mxfp4":
        b_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(B_scales),
            granularity="block",
            block_shape=(32,),
        )
        b_format = tensor_format("mxfp4", B.dtype, scale=b_scale)
        if weight_only:
            return format_signature(a=dense_tensor_format(A.dtype), b=b_format)
        a_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales),
            granularity="block",
            block_shape=(32,),
        )
        return format_signature(
            a=tensor_format("mxfp4", A.dtype, scale=a_scale), b=b_format
        )

    if quant == "mxfp8":
        if block_size is None:
            raise ValueError("mxfp8 format selection requires block_size")
        if B_scales is None:
            raise ValueError("mxfp8 format selection requires B_scales")
        b_scale = ScaleFormat(
            storage_dtype=B_scales.dtype,
            granularity="block",
            block_shape=tuple(block_size),
        )
        b_format = tensor_format("mxfp8", B.dtype, scale=b_scale)
        if weight_only:
            return format_signature(a=dense_tensor_format(A.dtype), b=b_format)
        a_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales),
            granularity="block",
            block_shape=tuple(block_size),
        )
        a_storage_dtype = fp8_storage_dtype if A_scales is None else A.dtype
        return format_signature(
            a=tensor_format("mxfp8", a_storage_dtype, scale=a_scale), b=b_format
        )

    raise ValueError(f"unknown quant {quant!r}; expected one of {SCALED_KINDS}")


def selection_log() -> list[dict]:
    """Return a copy of every distinct selection decision made in this process.

    Each entry has ``M``/``N``/``K``, the input dtype, and the winning
    ``kernel`` name. Used by tests and by the benchmark harness to *prove* which
    kernel was selected instead of inferring it from a timing delta.
    """
    with _lock:
        return list(_selection_log)


def dump_selection_evidence(path: str | None = None) -> str | None:
    """Write this process's selection decisions and kernel counters to JSON.

    Args:
        path: Destination prefix. Defaults to
            ``UNIEINFRA_KERNEL_SELECTION_DUMP``. The PID is appended, so every
            vLLM process (front end, EngineCore, worker) gets its own file and
            they never race.

    Returns:
        The path written, or ``None`` when no destination was configured.
    """
    target = (path or _SELECTION_DUMP).strip()
    if not target:
        return None
    import json

    counters: dict[str, dict[str, int]] = {}
    for module_name, attr in (
        ("unieinfra_kernel.ops.gemm.decode_gemv_triton", "gemv_stats"),
        ("unieinfra_kernel.ops.gemm.decode_gemv_fp8_triton", "fp8_gemv_stats"),
        ("unieinfra_kernel.ops.gemm.decode_gemv_nvfp4_triton", "nvfp4_gemv_stats"),
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            counters[module_name.rsplit(".", 1)[-1]] = getattr(module, attr)()

    out = f"{target}.{os.getpid()}.json"
    with open(out, "w") as handle:
        json.dump(
            {"pid": os.getpid(), "selection": selection_log(), "counters": counters},
            handle,
            indent=1,
        )
    return out


if _SELECTION_DUMP:  # pragma: no cover - opt-in diagnostics
    atexit.register(dump_selection_evidence)


def reset_dispatch_cache() -> None:
    """Drop the memoised per-shape selections (tests, or after registering)."""
    with _lock:
        _cache.clear()
        _scaled_cache.clear()
        _selection_log.clear()


def _resolve(
    M: int,
    N: int,
    K: int,
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> tuple[str, object, bool] | None:
    key = (M, N, K, in_dtype, out_dtype)
    try:
        return _cache[key]
    except KeyError:
        pass

    signature = format_signature(
        a=dense_tensor_format(in_dtype), b=dense_tensor_format(in_dtype)
    )
    try:
        selected = select_kernel(
            "gemm",
            "mm",
            signature,
            traits=shape_traits(M, N, K),
            # The concrete shape is what makes measurement-based selection
            # possible; the heuristic path ignores it. It is bucketed before it
            # reaches any cache key, so passing it costs nothing here.
            shape={"M": M, "N": N, "K": K},
        )
        resolved: tuple[str, object, bool] | None = (
            selected.name,
            selected,
            selected.name in _KERNELS_WITH_FUSED_BIAS,
        )
    except NoKernelFoundError:
        # No registered kernel claims this shape/dtype. Fall through to the
        # engine's own path; never an error, never a correctness change.
        resolved = None

    with _lock:
        _cache[key] = resolved
        entry = {
            "M": M,
            "N": N,
            "K": K,
            "dtype": str(in_dtype),
            "kernel": resolved[0] if resolved else "fallback:F.linear",
        }
        _selection_log.append(entry)
    if _SELECTION_DUMP:
        # Re-written on every *new* shape (a cache miss, so a handful of times
        # per process). atexit alone is not enough: vLLM's worker processes are
        # torn down without running exit handlers.
        dump_selection_evidence()
    if _LOG_SELECTION:
        logger.info(
            "[unieinfra_kernel] gemm.mm M=%d N=%d K=%d %s -> %s",
            M,
            N,
            K,
            in_dtype,
            entry["kernel"],
        )
    return resolved


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense linear through capability-based kernel selection.

    Semantically identical to ``torch.nn.functional.linear(x, weight, bias)``;
    only the implementation that computes it is chosen by the registry.

    Args:
        x: Activations ``[..., K]``.
        weight: Weights ``[N, K]``.
        bias: Optional ``[N]`` bias.

    Returns:
        ``[..., N]`` tensor in ``x``'s dtype.
    """
    if x.ndim != 2:
        flat = x.reshape(-1, x.shape[-1])
        out = linear(flat, weight, bias)
        return out.view(*x.shape[:-1], out.shape[-1])

    M, K = x.shape
    N = weight.shape[0]
    resolved = _resolve(M, N, K, x.dtype, x.dtype)
    if resolved is None:
        return torch.nn.functional.linear(x, weight, bias)

    _name, impl, fused_bias = resolved
    kwargs: dict[str, object] = {"alpha": None, "block_size": None}
    if fused_bias and bias is not None:
        kwargs["bias"] = bias
    out = impl(x, weight, None, None, x.dtype, **kwargs)
    if bias is not None and not fused_bias:
        out = out + bias.to(dtype=out.dtype)
    return out


# ---------------------------------------------------------------------------
# torch.library custom op
# ---------------------------------------------------------------------------
# Registering the façade as a custom op is what keeps torch.compile out of our
# selection logic: Dynamo sees one opaque call, Inductor cannot decompose it,
# and CUDA-graph capture records only the kernel the registry picked.

_CUSTOM_OP_AVAILABLE = hasattr(torch.library, "custom_op")

if _CUSTOM_OP_AVAILABLE:  # pragma: no branch - torch >= 2.4 everywhere we run

    @torch.library.custom_op("unieinfra::linear", mutates_args=())
    def _unieinfra_linear(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return linear(x, weight, bias)

    @_unieinfra_linear.register_fake
    def _(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        del bias
        return x.new_empty((*x.shape[:-1], weight.shape[0]))


def linear_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the custom op when available, else the plain Python façade.

    Args:
        x: Activations ``[..., K]``.
        weight: Weights ``[N, K]``.
        bias: Optional ``[N]`` bias.

    Returns:
        ``[..., N]`` tensor.
    """
    if _CUSTOM_OP_AVAILABLE:
        return torch.ops.unieinfra.linear(x, weight, bias)
    return linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Quantized-weight dispatch (fp8 / nvfp4)
# ---------------------------------------------------------------------------
# Same contract as the dense façade: selection is memoised per shape, the whole
# thing lives inside one opaque custom op, and *nothing matching* is not an
# error — it falls through to the host engine's own quantized kernel, so the
# absence of a first-party kernel for a shape is bit-for-bit "what vLLM would
# have done".
#
# The host engine supplies that fallback through :func:`register_engine_fallback`
# rather than being imported here, because ``unieinfra_kernel`` must stay
# vendor- and engine-neutral (AGENTS.md dependency boundaries).

# kind -> callable implementing the engine's own quantized GEMM.
_engine_fallbacks: dict[str, object] = {}

# (kind, M, N, K, in_dtype) -> (kernel_name, impl, fused_bias) or None.
_scaled_cache: dict[tuple, tuple[str, object, bool] | None] = {}


def register_engine_fallback(kind: str, fn: object) -> None:
    """Register the host engine's own GEMM for a quantized ``kind``.

    Called once by the engine integration (for vLLM, by
    ``unieinfra_vllm_plugin``) before any dispatch happens. The callable is
    invoked verbatim whenever no registered kernel claims a request, which is
    what makes "our dispatch declined" indistinguishable from stock behaviour.

    Args:
        kind: One of :data:`SCALED_KINDS`.
        fn: Engine GEMM. Its signature is fixed per kind — see
            :func:`linear_fp8` and :func:`linear_nvfp4`.
    """
    if kind not in SCALED_KINDS:
        raise ValueError(f"unknown quant kind {kind!r}; expected {SCALED_KINDS}")
    _engine_fallbacks[kind] = fn


def _resolve_scaled(
    kind: str,
    M: int,
    N: int,
    K: int,
    A: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor | None,
    block_size: list[int] | None = None,
) -> tuple[str, object, bool] | None:
    """Memoised selection for a quantized-weight request.

    The activation format is a *dispatch choice*, so this asks the registry for
    the weight-only signature (dense activation, quantized weight). If nothing
    claims it, the caller falls back to the engine, which performs its own
    online activation quantization and runs its own scaled GEMM.

    Returns:
        ``(kernel_name, impl, fused_bias)``, or ``None`` when no kernel matched.
    """
    # The scale's dtype and element count are part of the key because they are
    # part of the *signature*: a per-tensor and a per-channel fp8 weight scale
    # on the same (M, N, K) are two different requests and may resolve to two
    # different kernels.
    scale_key = None if B_scales is None else (B_scales.dtype, B_scales.numel())
    key = (kind, M, N, K, A.dtype, scale_key)
    try:
        return _scaled_cache[key]
    except KeyError:
        pass

    signature = gemm_format_signature(A, B, None, B_scales, kind, block_size)
    try:
        selected = select_kernel(
            "gemm",
            "mm",
            signature,
            traits=shape_traits(M, N, K),
            # The concrete shape is what makes measurement-based selection
            # possible; the heuristic path ignores it. It is bucketed before it
            # reaches any cache key, so passing it costs nothing here.
            shape={"M": M, "N": N, "K": K},
        )
        resolved: tuple[str, object, bool] | None = (
            selected.name,
            selected,
            selected.name in _KERNELS_WITH_FUSED_BIAS,
        )
    except NoKernelFoundError:
        resolved = None

    with _lock:
        _scaled_cache[key] = resolved
        entry = {
            "M": M,
            "N": N,
            "K": K,
            "dtype": str(A.dtype),
            "quant": kind,
            "kernel": resolved[0] if resolved else "fallback:engine",
        }
        _selection_log.append(entry)
    if _SELECTION_DUMP:
        dump_selection_evidence()
    if _LOG_SELECTION:
        logger.info(
            "[unieinfra_kernel] gemm.mm quant=%s M=%d N=%d K=%d %s -> %s",
            kind,
            M,
            N,
            K,
            A.dtype,
            entry["kernel"],
        )
    return resolved


def layer_is_claimable(
    kind: str,
    N: int,
    K: int,
    dtype: torch.dtype,
    *,
    max_m: int = 8,
) -> bool:
    """Whether *any* batch size could ever select one of our kernels for a layer.

    ``N`` and ``K`` are fixed when a layer is built; only ``M`` varies at
    runtime. So a projection whose width already fails the registry's trait gate
    (Qwen3-8B's ``gate_up_proj`` at ``N = 24576`` fails ``n_le_16384``) can never
    be served by us at any batch size, and wrapping it only adds cost: the
    dispatch op is opaque to Inductor, which measurably suppresses fusion around
    the GEMM (~2% at C >= 32 — see the control arm in the experiments log).

    Args:
        kind: One of :data:`SCALED_KINDS`.
        N: Output width of the projection.
        K: Reduction width.
        dtype: Activation dtype.
        max_m: Largest batch size to probe. The decode kernels all gate at
            ``m_le_8`` or tighter, so 8 is exhaustive for them.

    Returns:
        ``True`` when at least one ``M <= max_m`` resolves to a kernel.
    """
    a = torch.empty((1, K), dtype=dtype, device="meta")
    if kind == "nvfp4":
        b = torch.empty((N, K // 2), dtype=torch.uint8, device="meta")
        scales: torch.Tensor | None = torch.empty(
            (N, K // 16), dtype=torch.float8_e4m3fn, device="meta"
        )
    elif kind == "fp8":
        b = torch.empty((N, K), dtype=torch.float8_e4m3fn, device="meta")
        scales = torch.empty((1,), dtype=torch.float32, device="meta")
    else:  # pragma: no cover - no engine seam for mxfp8/mxfp4 yet
        return False

    signature = gemm_format_signature(a, b, None, scales, kind)
    m = 1
    while m <= max_m:
        try:
            # Explicitly heuristic: this is a load-time claimability probe over
            # every layer in the model, and it must never trigger tuning.
            select_kernel(
                "gemm",
                "mm",
                signature,
                traits=shape_traits(m, N, K),
                strategy=SelectionStrategy.HEURISTIC,
            )
            return True
        except NoKernelFoundError:
            pass
        m *= 2
    return False


def linear_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear against an FP8-E4M3 weight, through kernel selection.

    Args:
        x: Activations ``[..., K]`` in bf16/fp16 — *not* quantized. Whether the
            activation gets quantized is decided here, per selected kernel.
        weight: FP8 weight in the engine's storage orientation ``[K, N]``
            (``.t()`` of the logical ``[N, K]``), which is what vLLM's scaled-mm
            kernels keep.
        weight_scale: Per-tensor scalar or per-channel ``[N]`` fp32 scale.
        bias: Optional ``[N]`` bias.

    Returns:
        ``[..., N]`` tensor in ``x``'s dtype.
    """
    if x.ndim != 2:
        flat = x.reshape(-1, x.shape[-1])
        out = linear_fp8(flat, weight, weight_scale, bias)
        return out.view(*x.shape[:-1], out.shape[-1])

    M = x.shape[0]
    K, N = weight.shape
    b_nk = weight.t()  # logical [N, K]; free view, contiguous when weight is [K, N]
    resolved = _resolve_scaled("fp8", M, N, K, x, b_nk, weight_scale)
    if resolved is None:
        return _engine_fallbacks["fp8"](x, weight, weight_scale, bias)  # type: ignore[operator]

    _name, impl, fused_bias = resolved
    kwargs: dict[str, object] = {"alpha": None, "block_size": None}
    if fused_bias and bias is not None:
        kwargs["bias"] = bias
    out = impl(x, b_nk, None, weight_scale, x.dtype, **kwargs)
    if bias is not None and not fused_bias:
        out = out + bias.to(dtype=out.dtype)
    return out


def linear_nvfp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_linear: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear against an NVFP4 weight, through kernel selection.

    Args:
        x: Activations ``[..., K]`` in bf16/fp16 — *not* quantized.
        weight: Packed NVFP4 weight ``[N, K // 2]`` ``uint8``.
        weight_scale_linear: ``[N, K // 16]`` FP8-E4M3 block scales in linear
            layout (what a streaming kernel can index directly).
        weight_scale_swizzled: The same scales in the CUTLASS 128x4 layout, kept
            so the engine fallback is unchanged.
        weight_global_scale: Scalar fp32 weight global scale — the multiplier a
            weight-only kernel needs.
        input_global_scale_inv: Reciprocal activation global scale, used only by
            the engine fallback's online activation quantization.
        alpha: ``input_global_scale * weight_global_scale``, the engine
            fallback's fused scale.
        bias: Optional ``[N]`` bias.

    Returns:
        ``[..., N]`` tensor in ``x``'s dtype.
    """
    if x.ndim != 2:
        flat = x.reshape(-1, x.shape[-1])
        out = linear_nvfp4(
            flat,
            weight,
            weight_scale_linear,
            weight_scale_swizzled,
            weight_global_scale,
            input_global_scale_inv,
            alpha,
            bias,
        )
        return out.view(*x.shape[:-1], out.shape[-1])

    M, K = x.shape
    N = weight.shape[0]
    resolved = _resolve_scaled("nvfp4", M, N, K, x, weight, weight_scale_linear)
    if resolved is None:
        return _engine_fallbacks["nvfp4"](  # type: ignore[operator]
            x, weight, weight_scale_swizzled, input_global_scale_inv, alpha, bias
        )

    _name, impl, fused_bias = resolved
    kwargs: dict[str, object] = {
        "alpha": weight_global_scale,
        "block_size": None,
    }
    if fused_bias and bias is not None:
        kwargs["bias"] = bias
    out = impl(x, weight, None, weight_scale_linear, x.dtype, **kwargs)
    if bias is not None and not fused_bias:
        out = out + bias.to(dtype=out.dtype)
    return out


if _CUSTOM_OP_AVAILABLE:  # pragma: no branch - torch >= 2.4 everywhere we run

    @torch.library.custom_op("unieinfra::linear_fp8", mutates_args=())
    def _unieinfra_linear_fp8(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return linear_fp8(x, weight, weight_scale, bias)

    @_unieinfra_linear_fp8.register_fake
    def _(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        del weight_scale, bias
        return x.new_empty((*x.shape[:-1], weight.shape[1]))

    @torch.library.custom_op("unieinfra::linear_nvfp4", mutates_args=())
    def _unieinfra_linear_nvfp4(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_linear: torch.Tensor,
        weight_scale_swizzled: torch.Tensor,
        weight_global_scale: torch.Tensor,
        input_global_scale_inv: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return linear_nvfp4(
            x,
            weight,
            weight_scale_linear,
            weight_scale_swizzled,
            weight_global_scale,
            input_global_scale_inv,
            alpha,
            bias,
        )

    @_unieinfra_linear_nvfp4.register_fake
    def _(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_linear: torch.Tensor,
        weight_scale_swizzled: torch.Tensor,
        weight_global_scale: torch.Tensor,
        input_global_scale_inv: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        del weight_scale_linear, weight_scale_swizzled, weight_global_scale
        del input_global_scale_inv, alpha, bias
        return x.new_empty((*x.shape[:-1], weight.shape[0]))


def linear_fp8_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the fp8 custom op when available, else the plain Python façade."""
    if _CUSTOM_OP_AVAILABLE:
        return torch.ops.unieinfra.linear_fp8(x, weight, weight_scale, bias)
    return linear_fp8(x, weight, weight_scale, bias)


def linear_nvfp4_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_linear: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the nvfp4 custom op when available, else the plain Python façade."""
    if _CUSTOM_OP_AVAILABLE:
        return torch.ops.unieinfra.linear_nvfp4(
            x,
            weight,
            weight_scale_linear,
            weight_scale_swizzled,
            weight_global_scale,
            input_global_scale_inv,
            alpha,
            bias,
        )
    return linear_nvfp4(
        x,
        weight,
        weight_scale_linear,
        weight_scale_swizzled,
        weight_global_scale,
        input_global_scale_inv,
        alpha,
        bias,
    )
