# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Contract for the decode-shaped case of ``gemm.mm`` (small-M / narrow-N GEMV).

``gemm.mm`` is a pre-existing operator owned by ``tokenspeed-kernel``; this
module does **not** redefine it. What it freezes is the *decode slice* of that
operator — the shapes an autoregressive step actually issues:

    ``A [M, K]`` with ``M <= 8`` (one token per sequence) against a weight
    ``B [N, K]`` with a narrow ``N`` (``qkv_proj``, ``o_proj``, ...).

That slice is worth its own contract because it is a different kernel problem
from a prefill GEMM: arithmetic intensity is ~2 FLOP/byte, so the kernel is a
pure weight-streaming problem and the only figure of merit is achieved DRAM
bandwidth. See ``docs/experiments-log.md`` E26/E33.

The reference is ``F.linear`` — i.e. exactly what the incumbent ``torch_mm``
kernel calls — so the correctness oracle and the performance baseline are the
same function.
"""

from __future__ import annotations

from typing import Any

import torch
from unieinfra_kernel.boundary import (
    ArchVersion,
    CapabilityRequirement,
    FormatSignature,
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
    tensor_format,
)
from unieinfra_kernel.contracts import KernelContract, Tolerance
from unieinfra_kernel.selection.inputs import (
    set_benchmark_shapes,
    set_input_generator,
    set_shape_traits_fn,
)

__all__ = [
    "FAMILY",
    "MODE",
    "ACT_DTYPES",
    "DECODE_BATCH_GEMM",
    "DECODE_GEMV",
    "DECODE_GEMV_CAPABILITY",
    "FP8_CHANNEL_SCALE",
    "FP8_DTYPE",
    "FP8_TENSOR_SCALE",
    "NVFP4_BLOCK_SCALE",
    "W4A16_NVFP4_SIGNATURES",
    "W8A16_FP8_SIGNATURES",
    "decode_gemv_reference",
    "generate_gemm_inputs",
]

FAMILY = "gemm"
MODE = "mm"

# Same dense signatures the incumbent ``torch_mm`` declares, so a kernel
# fulfilling this contract is a drop-in candidate for the same requests.
_DENSE_SIGNATURES = format_signatures(
    ("a", "b"), "dense", {torch.bfloat16, torch.float16, torch.float32}
)

# Portable Triton: any CUDA-capable NVIDIA GPU from Ampere up, and AMD. The
# kernel body is vendor-neutral; the tuned config table is currently only
# populated for sm_120, which is why the registration below narrows this
# further (see ``decode_gemv_triton``).
#
# ``tensor_core:f16`` is the capability the version number was standing in for:
# this is a bf16/fp16 weight-streaming GEMV, so what it needs is half-precision
# tensor cores, which every NVIDIA part from sm_70 and every supported CDNA part
# advertises. Stating it that way is byte-identical to the bare
# ``min_arch_version`` on every part in the compat matrix (NVIDIA >= 8.0 all have
# it; gfx942/gfx950/gfx1250 all have it) while making the requirement decidable
# for silicon we have never seen -- which the version comparison alone is not.
DECODE_GEMV_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(8, 0),
    required_features=frozenset({"tensor_core:f16"}),
    vendors=frozenset({"nvidia", "amd"}),
)

# ---------------------------------------------------------------------------
# Scaled (quantized-weight) decode signatures
# ---------------------------------------------------------------------------
# The activation format on a decode step is a **dispatch choice**, not a
# property of the checkpoint: the weights are fp8 / nvfp4 on disk, but nothing
# forces the activation to be quantized too. At ``M <= 8`` the projection is a
# pure weight-streaming problem (E26), the activation is a few kilobytes, and
# quantizing it buys no bandwidth while costing a kernel launch and accuracy.
# So we publish **weight-only** signatures — dense bf16/fp16 activations against
# a scaled-fp8 or nvfp4 weight — alongside the symmetric W8A8/W4A4 ones that
# ``gemm.mm`` already speaks. ``unieinfra_kernel.dispatch`` offers the
# weight-only signature first and falls back to the symmetric one, which is
# where online activation quantization happens.

ACT_DTYPES = (torch.bfloat16, torch.float16)

FP8_DTYPE = torch.float8_e4m3fn

FP8_TENSOR_SCALE = ScaleFormat(storage_dtype=torch.float32, granularity="tensor")
FP8_CHANNEL_SCALE = ScaleFormat(storage_dtype=torch.float32, granularity="channel")

# NVFP4: one FP8-E4M3 scale per 16 contiguous ``K`` elements. Same
# ``block_shape=(16,)`` the ``gemm.mm`` nvfp4 branch builds, so a kernel
# registered here is a candidate for the identical request on either base.
NVFP4_BLOCK_SCALE = ScaleFormat(
    storage_dtype=torch.float8_e4m3fn,
    granularity="block",
    block_shape=(16,),
)

W8A16_FP8_SIGNATURES = frozenset(
    format_signature(
        a=dense_tensor_format(act_dtype),
        b=tensor_format("scaled-fp8", FP8_DTYPE, scale=scale),
    )
    for act_dtype in ACT_DTYPES
    for scale in (FP8_TENSOR_SCALE, FP8_CHANNEL_SCALE)
)

W4A16_NVFP4_SIGNATURES = frozenset(
    format_signature(
        a=dense_tensor_format(act_dtype),
        b=tensor_format("nvfp4", torch.uint8, scale=NVFP4_BLOCK_SCALE),
    )
    for act_dtype in ACT_DTYPES
)


def decode_gemv_reference(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference for the decode GEMV: ``A @ B.T (+ bias)``.

    Args:
        A: Activations ``[M, K]``.
        B: Weights ``[N, K]`` (row-major over ``K``, the ``torch.nn.Linear``
            layout).
        bias: Optional ``[N]`` bias added in the epilogue.

    Returns:
        ``[M, N]`` tensor in ``A``'s dtype — bit-identical to what the incumbent
        ``torch_mm`` kernel produces, because it is the same call.
    """
    return torch.nn.functional.linear(A, B, bias)


# Qwen3-8B decode shapes (hidden 4096). ``N`` is the projection's output width;
# ``K`` its input width. gate_up/down are included as *contrast* shapes: E26
# measured them already at 83-90% of peak, so they are where a decode GEMV must
# prove it does no harm, not where it is expected to win.
_QWEN3_8B_WORKLOADS = tuple(
    {"M": m, "N": n, "K": k, "label": label}
    for label, n, k in (
        ("qkv_proj", 6144, 4096),
        ("o_proj", 4096, 4096),
        ("gate_up_proj", 24576, 4096),
        ("down_proj", 4096, 12288),
    )
    for m in (1, 2, 4, 8)
)

DECODE_GEMV = KernelContract(
    family=FAMILY,
    mode=MODE,
    signatures=_DENSE_SIGNATURES,
    reference=decode_gemv_reference,
    # bf16 accumulate-in-fp32 against an fp32-accumulating reference: the only
    # difference is summation order over K, so a tight relative tolerance holds.
    tolerance=Tolerance(atol=1e-2, rtol=1e-2, required_matched_ratio=0.999),
    traits={"m_le_8": frozenset({True})},
    workloads=_QWEN3_8B_WORKLOADS,
    capability=DECODE_GEMV_CAPABILITY,
)


# ---------------------------------------------------------------------------
# Decode *batch* slice: the same projections at concurrency
# ---------------------------------------------------------------------------
# A step that decodes ``C`` sequences issues one row per sequence, so the same
# weight projections are called with ``M = C``. That is a different kernel
# problem from ``M <= 8``: at ``M >= 16`` the tensor cores are in play and the
# shape sits near the roofline knee, where E44 measured the incumbent losing a
# third of the machine to wave quantization rather than to any ceiling. The
# reference and the tolerance are unchanged -- it is the same arithmetic -- but
# the shape band, and therefore the kernel, is not.

_QWEN3_8B_BATCH_WORKLOADS = tuple(
    {"M": m, "N": n, "K": k, "label": label}
    for label, n, k in (
        ("qkv_proj", 6144, 4096),
        ("o_proj", 4096, 4096),
        ("gate_up_proj", 24576, 4096),
        ("down_proj", 4096, 12288),
    )
    for m in (32, 64, 128, 256)
)

DECODE_BATCH_GEMM = KernelContract(
    family=FAMILY,
    mode=MODE,
    signatures=_DENSE_SIGNATURES,
    reference=decode_gemv_reference,
    tolerance=Tolerance(atol=1e-2, rtol=1e-2, required_matched_ratio=0.999),
    traits={
        "m_min_16": frozenset({True}),
        "m_le_256": frozenset({True}),
        "n_le_32768": frozenset({True}),
    },
    workloads=_QWEN3_8B_BATCH_WORKLOADS,
    capability=DECODE_GEMV_CAPABILITY,
)


# ---------------------------------------------------------------------------
# Autotune input synthesis
# ---------------------------------------------------------------------------
# Autotuning cannot time a candidate without arguments, and the arguments are an
# operator property. Every ``gemm.mm`` kernel shares one calling convention --
# ``(A, B, A_scales, B_scales, out_dtype, *, alpha, block_size, bias, out)`` --
# which is exactly what makes them interchangeable, so one generator serves all
# of them. It reads the *format signature*, not the checkpoint, because the
# signature is what selection matched on: a weight-only fp8 request and a
# symmetric W8A8 request are two different signatures and need two different
# operand sets.


def _randn(
    shape: tuple[int, ...], dtype: torch.dtype, device: str, gen
) -> torch.Tensor:
    """Normal values materialised in ``dtype`` (via fp32 for narrow dtypes)."""
    values = torch.randn(*shape, dtype=torch.float32, device=device, generator=gen)
    return values.to(dtype)


def generate_gemm_inputs(
    *,
    format_signature: FormatSignature,
    shape: dict[str, Any],
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, Any]:
    """Synthesize one ``gemm.mm`` call's keyword arguments.

    Args:
        format_signature: The request's role-indexed signature. Role ``a`` is
            the activation, role ``b`` the weight; the weight's ``format``
            decides whether scales and a global ``alpha`` are produced.
        shape: Must contain integer ``M``, ``N`` and ``K``. Extra keys (such as
            the ``label`` carried by the contract's workloads) are ignored.
        device: Device to allocate on.
        seed: RNG seed. Autotuning varies it across rotated buffers, so the
            rotation is genuinely distinct memory rather than an alias.

    Returns:
        Kwargs accepted by every registered ``gemm.mm`` kernel.

    Raises:
        ValueError: If the shape is incomplete or the signature names a weight
            format this generator does not know how to build. Autotuning treats
            that as "not measurable" and keeps the heuristic answer.
    """
    try:
        M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"gemm.mm needs integer M/N/K, got {shape!r}") from exc

    a_format = format_signature.format_for("a")
    b_format = format_signature.format_for("b")
    if a_format is None or b_format is None:
        raise ValueError(f"gemm.mm signature needs roles a and b: {format_signature}")
    if a_format.format != "dense":
        # Symmetric W8A8 / W4A4 requests exist in the vocabulary but no
        # first-party kernel claims them yet, so there is nothing to tune
        # between. Refusing here is honest; inventing a quantized activation
        # would tune a path that never runs.
        raise ValueError(
            f"gemm.mm autotune input generator supports a dense activation only, "
            f"got a={a_format}"
        )

    gen = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    gen.manual_seed(seed)

    act_dtype = a_format.storage_dtype
    inputs: dict[str, Any] = {
        "A": _randn((M, K), act_dtype, device, gen),
        "A_scales": None,
        "out_dtype": act_dtype,
        "alpha": None,
        "block_size": None,
        "bias": None,
        "out": None,
    }

    if b_format.format == "dense":
        inputs["B"] = _randn((N, K), b_format.storage_dtype, device, gen)
        inputs["B_scales"] = None
        return inputs

    if b_format.format == "scaled-fp8":
        scale = b_format.scale
        inputs["B"] = (
            _randn((N, K), torch.float32, device, gen)
            .clamp(-448.0, 448.0)
            .to(b_format.storage_dtype)
        )
        scale_shape = (1,) if (scale and scale.granularity == "tensor") else (N, 1)
        inputs["B_scales"] = torch.full(
            scale_shape,
            0.05,
            dtype=(scale.storage_dtype if scale else torch.float32),
            device=device,
        )
        return inputs

    if b_format.format == "nvfp4":
        if K % 16 != 0:
            raise ValueError(f"nvfp4 needs K divisible by 16, got K={K}")
        # Packed uint8: two fp4 nibbles per byte, one fp8-e4m3 scale per 16 K.
        inputs["B"] = torch.randint(
            0, 256, (N, K // 2), dtype=torch.uint8, device=device, generator=gen
        )
        inputs["B_scales"] = (
            torch.rand((N, K // 16), dtype=torch.float32, device=device, generator=gen)
            .add(0.5)
            .to(b_format.scale.storage_dtype if b_format.scale else torch.float8_e4m3fn)
        )
        inputs["alpha"] = torch.tensor(0.05, dtype=torch.float32, device=device)
        return inputs

    raise ValueError(
        f"no gemm.mm input synthesis for weight format {b_format.format!r}"
    )


def _gemm_shape_traits(shape: dict[str, Any]) -> dict[str, Any] | None:
    """Delegate to the dispatch façade's canonical trait vocabulary.

    Imported lazily so this contract module stays importable without the
    dispatch layer (and so there is no import cycle: ``dispatch`` imports the
    selection boundary, which is what registers this hook's consumer).
    """
    if not all(isinstance(shape.get(k), int) for k in ("M", "N", "K")):
        return None
    from unieinfra_kernel.dispatch import shape_traits

    return shape_traits(int(shape["M"]), int(shape["N"]), int(shape["K"]))


set_input_generator(FAMILY, MODE, generate_gemm_inputs)
set_shape_traits_fn(FAMILY, MODE, _gemm_shape_traits)
# The contract's own workloads are what startup tuning measures over when the
# caller does not name shapes: they are the Qwen3-8B decode projections plus the
# two contrast shapes a decode GEMV has to prove it does no harm on. Both decode
# bands are listed -- batch-1 (``M <= 8``) and batch decode (``M = 32..256``) --
# because ``gemm.mm`` has one benchmark-shape list and both contracts' kernels
# are tuned against it.
set_benchmark_shapes(
    FAMILY,
    MODE,
    list(DECODE_GEMV.workloads) + list(DECODE_BATCH_GEMM.workloads),
)
