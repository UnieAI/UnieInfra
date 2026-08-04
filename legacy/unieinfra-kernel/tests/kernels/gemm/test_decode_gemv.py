# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Correctness and gating tests for the first-party decode GEMV.

Two halves, mirroring ``tests/kernels/activation/test_discovery.py``:

* **CPU half** — registration, the trait gate, and the "never displaces the
  incumbent outside its slice" invariant. Selection is pure Python, so these run
  everywhere and are the tests that protect the existing path.
* **GPU half** — numerics against ``F.linear`` (the same function the incumbent
  ``torch_mm`` calls) across the Qwen3-8B decode shapes, batch sizes, bias, an
  ``out=`` buffer, a strided output view, and the split-K path.

Shape coverage follows vLLM's ``tests/kernels/test_bf16_skinny_gemm.py``
structure: every supported shape crossed with every supported M, plus a strided
output case.
"""

from __future__ import annotations

import pytest
import torch

# Importing the package triggers discovery -> the kernel self-registers.
import unieinfra_kernel  # noqa: F401
from tokenspeed_kernel.platform import (
    ArchVersion,
    PlatformInfo,
    _get_cuda_sm_features,
)
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    select_kernel,
    spec_matches_shape_traits,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from unieinfra_kernel.contracts.gemm import DECODE_GEMV, decode_gemv_reference
from unieinfra_kernel.ops.gemm.decode_gemv_triton import (
    MAX_DECODE_M,
    GemvConfig,
    _is_profitable,
    decode_gemv,
    select_gemv_config,
    supports_decode_gemv,
    unieinfra_decode_gemv_triton,
)

KERNEL_NAME = "unieinfra_decode_gemv_triton"

_SIG = format_signature(
    a=dense_tensor_format(torch.bfloat16), b=dense_tensor_format(torch.bfloat16)
)

# Qwen3-8B decode projections (hidden 4096).
SHAPES = [(6144, 4096), (4096, 4096), (4096, 12288)]
MS = [1, 2, 4, 8]


def _platform(major: int, minor: int) -> PlatformInfo:
    """Synthesize a minimal NVIDIA platform at the given arch (no GPU needed)."""
    return PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(major, minor),
        device_name=f"synthetic-sm_{major}{minor}",
        device_count=1,
        total_memory=0,
        memory_bandwidth=0.0,
        sm_count=188,
        max_threads_per_sm=0,
        max_shared_memory_per_sm=0,
        # Derived from the real detector, never hand-copied (E30).
        sm_features=_get_cuda_sm_features(ArchVersion(major, minor)),
    )


SM120 = _platform(12, 0)
SM75 = _platform(7, 5)


def _ensure_incumbent() -> None:
    """Make sure ``torch_mm`` (the kernel we must never displace) is registered.

    On a GPU box ``tokenspeed_kernel.numerics.reference.gemm`` imports and
    registers the real one. On the CPU-only test box that module pulls in the
    GPU toolchain, so a byte-identical stand-in is registered instead: same
    name, same signatures, same ``PORTABLE + 3`` priority, no traits. The point
    of these tests is the *selection* outcome, which depends only on those.
    """
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures

    if KernelRegistry.get().get_by_name("torch_mm") is not None:
        return
    try:
        import tokenspeed_kernel.numerics.reference.gemm  # noqa: F401

        if KernelRegistry.get().get_by_name("torch_mm") is not None:
            return
    except Exception:  # noqa: BLE001 - no GPU toolchain on this box
        pass

    @register_kernel(
        "gemm",
        "mm",
        name="torch_mm",
        solution="reference",
        signatures=format_signatures(
            ("a", "b"), "dense", {torch.bfloat16, torch.float16, torch.float32}
        ),
        traits={},
        priority=Priority.PORTABLE + 3,
        tags={"determinism", "portability"},
    )
    def _torch_mm(A, B, A_scales, B_scales, out_dtype, **kwargs):
        return torch.nn.functional.linear(A, B, kwargs.get("bias"))


_ensure_incumbent()


def _decode_traits(M: int, N: int, K: int) -> dict[str, object]:
    """The trait vocabulary a real ``gemm.mm`` request carries.

    Delegates to the dispatch façade rather than hand-listing the buckets this
    kernel happens to gate on. A partial dict is not a smaller test, it is a
    *different* one: ``spec_matches_traits`` only filters on traits the request
    actually supplies, so omitting a bucket makes every kernel that gates on it
    look unconditionally eligible. That is how a sibling kernel registered later
    (E45's decode-batch GEMM declares ``m_min_16``) would silently become a
    candidate here at ``M = 128`` and quietly invert the "never displaces the
    incumbent" assertion below.
    """
    from unieinfra_kernel.dispatch import shape_traits

    return shape_traits(M, N, K)


# ---------------------------------------------------------------------------
# CPU: registration + gating
# ---------------------------------------------------------------------------


def test_kernel_is_discovered_and_registered() -> None:
    spec = KernelRegistry.get().get_by_name(KERNEL_NAME)
    assert spec is not None, "discovery did not register the decode GEMV"
    assert spec.origin == "first_party"
    assert (spec.family, spec.mode) == DECODE_GEMV.operator
    assert KernelRegistry.get().get_impl(KERNEL_NAME) is not None


def test_wins_selection_for_decode_shapes() -> None:
    """It is actually reachable for the qkv_proj / o_proj decode requests."""
    for N, K in SHAPES:
        selected = select_kernel(
            "gemm", "mm", _SIG, traits=_decode_traits(1, N, K), platform=SM120
        )
        assert selected.name == KERNEL_NAME, f"not selected for N={N} K={K}"


@pytest.mark.parametrize(
    "M,N,K,reason,falls_back_to",
    [
        # Mid-M is not the incumbent's any more: the decode-*batch* GEMM (E45)
        # owns 16 <= M <= 256. What matters for *this* kernel's safety argument
        # is unchanged -- it is not selected -- so the expected winner is named
        # per case rather than assumed to be torch_mm.
        (
            16,
            4096,
            4096,
            "decode batch, not a GEMV",
            "unieinfra_decode_batch_gemm_triton",
        ),
        (
            128,
            4096,
            4096,
            "decode batch, not a GEMV",
            "unieinfra_decode_batch_gemm_triton",
        ),
        (512, 4096, 4096, "prefill-shaped M", "torch_mm"),
        (1, 24576, 4096, "wide output (gate_up_proj)", "torch_mm"),
        (1, 151936, 4096, "lm_head", "torch_mm"),
    ],
)
def test_never_displaces_incumbent_outside_its_slice(
    M: int, N: int, K: int, reason: str, falls_back_to: str
) -> None:
    """The whole safety argument: outside (small M, narrow N) we are invisible."""
    selected = select_kernel(
        "gemm", "mm", _SIG, traits=_decode_traits(M, N, K), platform=SM120
    )
    assert selected.name != KERNEL_NAME, f"should not have been selected for {reason}"
    assert selected.name == falls_back_to


def test_capability_gate_excludes_pre_sm80() -> None:
    selected = select_kernel(
        "gemm", "mm", _SIG, traits=_decode_traits(1, 4096, 4096), platform=SM75
    )
    assert selected.name != KERNEL_NAME


def test_trait_gate_is_declarative() -> None:
    """The gate lives on the spec, not only in the wrapper's runtime check."""
    spec = KernelRegistry.get().get_by_name(KERNEL_NAME)
    assert spec_matches_traits(spec, _decode_traits(1, 4096, 4096))
    assert not spec_matches_traits(spec, _decode_traits(9, 4096, 4096))
    assert not spec_matches_traits(spec, _decode_traits(1, 24576, 4096))
    assert spec_matches_shape_traits(spec, {"M": 1, "N": 4096, "K": 4096})
    assert not spec_matches_shape_traits(spec, {"M": 64, "N": 4096, "K": 4096})
    assert not spec_matches_shape_traits(spec, {"M": 1, "N": 32768, "K": 4096})


def test_new_shape_traits_do_not_disturb_existing_kernels() -> None:
    """No pre-existing kernel declares the new buckets, so none can be filtered."""
    registry = KernelRegistry.get()
    # The quantized decode GEMVs are siblings of this one: they declare the
    # decode buckets on purpose and are excluded for the same reason.
    ours = {
        KERNEL_NAME,
        "unieinfra_decode_gemv_fp8_triton",
        "unieinfra_decode_gemv_nvfp4_triton",
    }
    for spec in registry.list_kernels(family="gemm"):
        if spec.name in ours:
            continue
        for bucket in ("m_le_8", "m_le_4", "m_le_2", "n_le_16384"):
            assert bucket not in spec.traits


def test_config_heuristic_keeps_the_grid_large() -> None:
    """The whole point: narrow N must still produce thousands of blocks."""
    for N, K in SHAPES:
        cfg = select_gemv_config(1, N, K)
        blocks = -(-N // cfg.block_n)
        assert blocks >= 2 * 188, f"grid {blocks} too small for N={N}"
        assert cfg.block_n * cfg.block_k <= 8192


def test_unsupported_requests_are_rejected_by_the_runtime_guard() -> None:
    a = torch.randn(MAX_DECODE_M + 1, 64)
    b = torch.randn(128, 64)
    assert not supports_decode_gemv(a, b)  # M too large (and CPU tensors)


@pytest.mark.parametrize(
    "M,expected", [(1, True), (2, True), (4, False), (8, False), (64, False)]
)
def test_profitability_rule_matches_the_measured_region(M: int, expected: bool) -> None:
    """The runtime guard only claims batch sizes where E33 measured a win."""
    assert _is_profitable(M, 6144, 4096) is expected
    assert _is_profitable(M, 4096, 4096) is expected


# ---------------------------------------------------------------------------
# GPU: numerics
# ---------------------------------------------------------------------------

gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="decode GEMV numerics need a GPU"
)


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    ).item()


@gpu
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_reference(n: int, k: int, m: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(n, k, dtype=dtype, device="cuda")
    out = decode_gemv(a, b)
    assert out.shape == (m, n) and out.dtype == dtype
    assert _rel_err(out, decode_gemv_reference(a, b)) < DECODE_GEMV.tolerance.rtol


@gpu
@pytest.mark.parametrize("m", MS)
def test_bias_is_fused(m: int) -> None:
    torch.manual_seed(0)
    a = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(6144, 4096, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(6144, dtype=torch.bfloat16, device="cuda")
    out = decode_gemv(a, b, bias=bias)
    assert _rel_err(out, decode_gemv_reference(a, b, bias)) < 1e-2


@gpu
def test_out_buffer_and_strided_view() -> None:
    """``mm`` may hand us a column slice of a wider buffer (stride(0) > N)."""
    torch.manual_seed(0)
    a = torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    ref = decode_gemv_reference(a, b)

    dst = torch.empty(2, 4096, dtype=torch.bfloat16, device="cuda")
    assert decode_gemv(a, b, out=dst).data_ptr() == dst.data_ptr()
    assert _rel_err(dst, ref) < 1e-2

    wide = torch.zeros(2, 5000, dtype=torch.bfloat16, device="cuda")
    view = wide[:, 100:4196]
    decode_gemv(a, b, out=view)
    assert _rel_err(view, ref) < 1e-2
    assert wide[:, :100].abs().max().item() == 0.0  # nothing scribbled outside


@gpu
@pytest.mark.parametrize("split_k", [2, 4, 8])
def test_split_k_path(split_k: int) -> None:
    torch.manual_seed(0)
    a = torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    cfg = GemvConfig(block_n=4, block_k=256, split_k=split_k, num_warps=4)
    out = decode_gemv(a, b, bias=bias, config=cfg)
    assert _rel_err(out, decode_gemv_reference(a, b, bias)) < 1e-2


@gpu
@pytest.mark.parametrize("m", MS)
def test_registered_entry_point_matches_reference(m: int) -> None:
    """Exercise the exact callable ``gemm.mm`` dispatch invokes."""
    torch.manual_seed(0)
    a = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(6144, 4096, dtype=torch.bfloat16, device="cuda")
    out = unieinfra_decode_gemv_triton(a, b, None, None, torch.bfloat16)
    assert _rel_err(out, decode_gemv_reference(a, b)) < 1e-2


@gpu
def test_falls_back_instead_of_failing_on_unsupported_input() -> None:
    """A shape the fast path rejects must still return the right answer."""
    torch.manual_seed(0)
    a = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")  # M > 8
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    out = unieinfra_decode_gemv_triton(a, b, None, None, torch.bfloat16)
    assert _rel_err(out, decode_gemv_reference(a, b)) < 1e-2

    nc = torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda").t().t()[:, ::2]
    assert not supports_decode_gemv(nc, b[:, ::2])


@gpu
def test_cuda_graph_capturable() -> None:
    """Decode runs captured (E7/E29): the kernel must not sync or autotune."""
    torch.manual_seed(0)
    a = torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(6144, 4096, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(1, 6144, dtype=torch.bfloat16, device="cuda")
    ref = decode_gemv_reference(a, b)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            decode_gemv(a, b, out=out)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_gemv(a, b, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 1e-2
