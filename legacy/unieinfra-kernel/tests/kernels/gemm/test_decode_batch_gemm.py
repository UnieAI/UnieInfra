# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Correctness and gating tests for the first-party decode-batch GEMM (E45).

Two halves, mirroring ``test_decode_gemv.py``:

* **CPU half** — registration, the trait gate, the "gate equals profitability
  rule" invariant that E38 paid for, and the guarantee that this kernel and the
  batch-1 decode GEMV occupy disjoint bands and never compete.
* **GPU half** — numerics against ``F.linear`` (the same function the incumbent
  ``torch_mm`` calls) across the Qwen3-8B decode projections and the decode
  concurrencies a served engine runs at, plus bias, ``out=``, a strided output
  view, the split-K path, and CUDA-graph capture.
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
from unieinfra_kernel.contracts.gemm import DECODE_BATCH_GEMM, decode_gemv_reference
from unieinfra_kernel.dispatch import shape_traits
from unieinfra_kernel.ops.gemm.decode_batch_gemm_triton import (
    MAX_BATCH_M,
    MIN_BATCH_M,
    BatchGemmConfig,
    _is_profitable,
    decode_batch_gemm,
    select_batch_gemm_config,
    supports_decode_batch_gemm,
    unieinfra_decode_batch_gemm_triton,
    wave_utilisation,
)

KERNEL_NAME = "unieinfra_decode_batch_gemm_triton"

_SIG = format_signature(
    a=dense_tensor_format(torch.bfloat16), b=dense_tensor_format(torch.bfloat16)
)

# Qwen3-8B decode projections (hidden 4096), as (N, K).
SHAPES = [(24576, 4096), (4096, 12288), (6144, 4096), (4096, 4096)]
# The decode concurrencies this kernel is tuned for.
MS = [32, 64, 128, 256]


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
    """Register a stand-in for ``torch_mm`` when the GPU toolchain is absent.

    Same name, signatures and ``PORTABLE + 3`` priority as the real one, which
    is all the *selection* outcome under test depends on.
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


# ---------------------------------------------------------------------------
# CPU: registration + gating
# ---------------------------------------------------------------------------


def test_kernel_is_discovered_and_registered() -> None:
    spec = KernelRegistry.get().get_by_name(KERNEL_NAME)
    assert spec is not None, "discovery did not register the decode-batch GEMM"
    assert spec.origin == "first_party"
    assert (spec.family, spec.mode) == DECODE_BATCH_GEMM.operator
    assert KernelRegistry.get().get_impl(KERNEL_NAME) is not None


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_wins_selection_for_decode_batch_shapes(n: int, k: int, m: int) -> None:
    """It is actually reachable for the projections E44 measured."""
    selected = select_kernel(
        "gemm", "mm", _SIG, traits=shape_traits(m, n, k), platform=SM120
    )
    assert selected.name == KERNEL_NAME, f"not selected for M={m} N={n} K={k}"


@pytest.mark.parametrize(
    "M,N,K,reason",
    [
        (1, 4096, 4096, "batch-1 decode belongs to the GEMV"),
        (8, 6144, 4096, "still the GEMV's band"),
        (512, 4096, 4096, "prefill-shaped M"),
        (2048, 24576, 4096, "prefill"),
        (128, 151936, 4096, "lm_head: N is not the decode projection shape"),
    ],
)
def test_never_displaces_incumbent_outside_its_band(
    M: int, N: int, K: int, reason: str
) -> None:
    """Outside 16 <= M <= 256 we are invisible to selection."""
    selected = select_kernel(
        "gemm", "mm", _SIG, traits=shape_traits(M, N, K), platform=SM120
    )
    assert selected.name != KERNEL_NAME, f"should not have been selected for {reason}"


def test_capability_gate_excludes_pre_sm80() -> None:
    selected = select_kernel(
        "gemm", "mm", _SIG, traits=shape_traits(128, 4096, 4096), platform=SM75
    )
    assert selected.name != KERNEL_NAME


def test_declarative_gate_equals_the_profitability_rule() -> None:
    """The E38 invariant, checked over the whole range CUDA graphs capture.

    ``torch.compile`` + CUDA-graph capture walks ``M = 1 ... 512`` at engine
    construction. If the declarative trait admitted a shape the runtime rule
    then declined, the decline would happen *during capture* — so the two must
    agree for every M in that walk, not merely at the shapes we benchmarked.
    """
    spec = KernelRegistry.get().get_by_name(KERNEL_NAME)
    # lm_head is included on purpose: it is the shape whose N bound the gate has
    # to express, and it is reached by the same capture walk.
    for N, K in SHAPES + [(151936, 4096)]:
        for M in range(1, 513):
            admitted = spec_matches_traits(spec, shape_traits(M, N, K))
            assert admitted is _is_profitable(
                M, N, K
            ), f"trait gate and profitability rule disagree at M={M} N={N} K={K}"


def test_trait_gate_is_declarative() -> None:
    """The gate lives on the spec, not only in the wrapper's runtime check.

    ``spec_matches_traits`` is the predicate the *dispatch* path uses: it reads
    the request's trait dict, so it needs no per-trait table and is identical
    whichever selection backend is active. ``spec_matches_shape_traits`` is the
    autotune-side filter and does carry a hardcoded bucket table, which lives in
    the active backend — so it is only asserted here on the alignment buckets
    every backend has always known about.
    """
    spec = KernelRegistry.get().get_by_name(KERNEL_NAME)
    assert spec_matches_traits(spec, shape_traits(128, 24576, 4096))
    assert not spec_matches_traits(spec, shape_traits(8, 24576, 4096))
    assert not spec_matches_traits(spec, shape_traits(257, 24576, 4096))
    # A projection whose N is not a multiple of 64 is not ours.
    assert not spec_matches_traits(spec, shape_traits(128, 4100, 4096))
    # Neither is lm_head, whose N is the vocabulary rather than a projection.
    assert not spec_matches_traits(spec, shape_traits(128, 151936, 4096))
    assert spec_matches_shape_traits(spec, {"M": 128, "N": 4096, "K": 4096})
    assert not spec_matches_shape_traits(spec, {"M": 128, "N": 4096, "K": 4100})


def test_bands_are_disjoint_from_the_batch1_decode_gemv() -> None:
    """The two decode kernels can never both be candidates for one request."""
    registry = KernelRegistry.get()
    batch = registry.get_by_name(KERNEL_NAME)
    gemv = registry.get_by_name("unieinfra_decode_gemv_triton")
    assert gemv is not None
    for M in range(1, 513):
        traits = shape_traits(M, 4096, 4096)
        both = spec_matches_traits(batch, traits) and spec_matches_traits(gemv, traits)
        assert not both, f"both decode kernels claim M={M}"


def test_fused_bias_is_declared_to_the_dispatch_facade() -> None:
    """The façade must know we apply bias ourselves, or it would add it twice."""
    from unieinfra_kernel.dispatch import _KERNELS_WITH_FUSED_BIAS

    assert KERNEL_NAME in _KERNELS_WITH_FUSED_BIAS


def test_every_selected_config_is_launchable() -> None:
    """A config the table or the heuristic returns must not fail at launch.

    Three ways it could, all of them at run time and none of them caught by any
    other test: a ``BLOCK_K * SPLIT_K`` that does not divide ``K`` silently
    drops part of the reduction, an MMA tile below 16 rows does not compile, and
    a pipelined tile past the shared-memory budget fails in the driver. The
    third is the reason the heuristic backs ``BLOCK_K`` off in a loop.
    """
    from unieinfra_kernel.ops.gemm.decode_batch_gemm_triton import (
        _SMEM_PER_SM,
        _smem_bytes,
    )

    # The shipped shapes, plus shapes deliberately outside the tuned table so
    # the heuristic path is exercised too (a second model's projections).
    extra = [(5120, 5120), (13824, 5120), (1536, 8192)]
    for N, K in SHAPES + extra:
        for M in (16, 32, 33, 64, 128, 200, 256):
            cfg = select_batch_gemm_config(M, N, K)
            assert cfg.block_m >= 16, f"M={M} N={N} K={K}: MMA tile too short"
            assert (
                K % (cfg.block_k * cfg.split_k) == 0
            ), f"M={M} N={N} K={K}: BLOCK_K*SPLIT_K does not divide K"
            assert (
                _smem_bytes(cfg) <= _SMEM_PER_SM
            ), f"M={M} N={N} K={K}: {_smem_bytes(cfg)} B of shared memory"


def test_heuristic_prefers_the_tile_over_the_grid() -> None:
    """The refuted-hypothesis guard.

    The intuitive fix for E44's wave quantization is "more, smaller CTAs until
    the rounds divide evenly". The sweep measured that as *slower* than a wide
    tile that leaves half the SMs idle, so the heuristic must not regress to
    maximising :func:`wave_utilisation`. Asserted on ``gate_up``, where the two
    answers are furthest apart: 96 fat tiles beat 3072 thin ones by 3%.
    """
    cfg = select_batch_gemm_config(128, 24576, 4096)
    assert cfg.block_n * cfg.block_k >= 8192, "regressed to a thin tile"
    util, _tiles, _per_sm = wave_utilisation(128, 24576, 4096, cfg, 188)
    # Explicitly NOT a high number: the winning config scores ~0.51.
    assert util < 0.9


def test_wave_utilisation_reproduces_the_e44_incumbent_loss() -> None:
    """The model must reproduce the number the counters actually showed.

    E44/E45 measured the incumbent ``gate_up`` kernel — CUTLASS calls it
    ``64x256_32x4`` — launching **384 CTAs of 128 threads, one resident per SM**
    on a 188-SM part, with ``sm__throughput 62.1%`` against ``tensor pipe
    90.7%`` active: a ratio of 0.685. Three rounds of 188/188/8 is
    ``384 / (3 * 188) = 68.1%``.

    The CTA tile asserted here is ``64x128``, which is what the *measured* grid
    implies (``ceil(128/64) * ceil(24576/128) = 384``) and not what the CUTLASS
    kernel name reads as; the name's ``256`` does not correspond to the CTA's N
    extent, and there is no split-K reduction launch to explain the factor of
    two either. Deriving the geometry from the counter is the checkable move.
    """
    incumbent = BatchGemmConfig(
        block_m=64, block_n=128, block_k=32, num_stages=4, num_warps=4
    )
    # Residency is the profiler's, not an estimate: ncu reported
    # launch__occupancy_limit_shared_mem = 1 for this kernel.
    util, tiles, per_sm = wave_utilisation(
        128, 24576, 4096, incumbent, 188, ctas_per_sm=1
    )
    assert tiles == 384, "does not reproduce the measured grid"
    assert per_sm == 1
    assert 0.66 < util < 0.70, f"modelled {util:.3f}, counters implied 0.685"


def test_unsupported_requests_are_rejected_by_the_runtime_guard() -> None:
    a = torch.randn(128, 4096)
    b = torch.randn(4096, 4096)
    assert not supports_decode_batch_gemm(a, b)  # CPU tensors
    assert not supports_decode_batch_gemm(
        torch.randn(MIN_BATCH_M - 1, 4096), torch.randn(4096, 4096)
    )
    assert not supports_decode_batch_gemm(
        torch.randn(MAX_BATCH_M + 1, 4096), torch.randn(4096, 4096)
    )


@pytest.mark.parametrize(
    "M,expected",
    [
        (1, False),
        (8, False),
        (15, False),
        (16, True),
        (128, True),
        (256, True),
        (257, False),
        (512, False),
    ],
)
def test_profitability_rule_matches_the_measured_band(M: int, expected: bool) -> None:
    assert _is_profitable(M, 24576, 4096) is expected


# ---------------------------------------------------------------------------
# GPU: numerics
# ---------------------------------------------------------------------------

gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="decode-batch GEMM numerics need a GPU"
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
    out = decode_batch_gemm(a, b)
    assert out.shape == (m, n) and out.dtype == dtype
    assert _rel_err(out, decode_gemv_reference(a, b)) < DECODE_BATCH_GEMM.tolerance.rtol


@gpu
@pytest.mark.parametrize("m", [16, 33, 64, 127, 200, 256])
def test_matches_reference_at_ragged_batch_sizes(m: int) -> None:
    """A served step does not politely arrive at a power of two."""
    torch.manual_seed(0)
    a = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(6144, 4096, dtype=torch.bfloat16, device="cuda")
    out = decode_batch_gemm(a, b)
    assert _rel_err(out, decode_gemv_reference(a, b)) < 1e-2


@gpu
def test_bias_is_fused() -> None:
    torch.manual_seed(0)
    a = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(6144, 4096, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(6144, dtype=torch.bfloat16, device="cuda")
    out = decode_batch_gemm(a, b, bias=bias)
    assert _rel_err(out, decode_gemv_reference(a, b, bias)) < 1e-2


@gpu
def test_out_buffer_and_strided_view() -> None:
    torch.manual_seed(0)
    a = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    ref = decode_gemv_reference(a, b)

    buf = torch.empty(64, 4096, dtype=torch.bfloat16, device="cuda")
    got = decode_batch_gemm(a, b, out=buf)
    assert got.data_ptr() == buf.data_ptr()
    assert _rel_err(buf, ref) < 1e-2

    # A row-strided destination, as a fused QKV split would hand us.
    wide = torch.empty(64, 8192, dtype=torch.bfloat16, device="cuda")
    view = wide[:, :4096]
    decode_batch_gemm(a, b, out=view)
    assert _rel_err(view, ref) < 1e-2


@gpu
def test_split_k_path_matches_and_is_deterministic() -> None:
    """Split-K writes disjoint fp32 slices, so repeated runs must be bit-equal."""
    torch.manual_seed(0)
    a = torch.randn(128, 12288, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 12288, dtype=torch.bfloat16, device="cuda")
    cfg = BatchGemmConfig(
        block_m=128, block_n=64, block_k=64, split_k=4, num_warps=4, num_stages=3
    )
    first = decode_batch_gemm(a, b, config=cfg)
    second = decode_batch_gemm(a, b, config=cfg)
    assert torch.equal(first, second), "split-K result is not reproducible"
    assert _rel_err(first, decode_gemv_reference(a, b)) < 1e-2


@gpu
def test_is_cuda_graph_capturable() -> None:
    """No host sync, no autotune, no allocation surprises during capture."""
    torch.manual_seed(0)
    a = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(6144, 4096, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(128, 6144, dtype=torch.bfloat16, device="cuda")
    cfg = select_batch_gemm_config(128, 6144, 4096)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            decode_batch_gemm(a, b, out=out, config=cfg)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_batch_gemm(a, b, out=out, config=cfg)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert _rel_err(out, decode_gemv_reference(a, b)) < 1e-2
    del graph


@gpu
def test_registered_entry_point_delegates_outside_its_band() -> None:
    """A declined shape returns exactly what the incumbent would have."""
    torch.manual_seed(0)
    a = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    got = unieinfra_decode_batch_gemm_triton(a, b, None, None, torch.bfloat16)
    assert torch.equal(got, decode_gemv_reference(a, b))
