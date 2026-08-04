# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Tests for measurement-based kernel selection.

The GPU-free tests cover the parts that decide *what* gets measured and *what
is done with the answer* — bucketing, cache keying, the freeze, the fail-safe
drop, objective handling and pins. The two GPU tests check the thing that only
hardware can answer: that timing real kernels on real shapes reproduces the
decisions E33/E34 established.
"""

from __future__ import annotations

import pytest
import torch
from unieinfra_kernel.selection import selection as sel
from unieinfra_kernel.selection.autotune import (
    AutotuneCache,
    AutotuneParams,
    Measurement,
    shape_bucket,
)

# These tests exercise the *standalone* selection core directly, so they import
# from it rather than through `unieinfra_kernel.boundary`. On a box where
# `tokenspeed_kernel` is importable the boundary resolves to that package's
# registry, which is a different singleton from the one
# `unieinfra_kernel.selection.selection` drives -- registering into one and
# selecting from the other would silently find no candidates.
from unieinfra_kernel.selection.platform import (
    ArchVersion,
    CapabilityRequirement,
    PlatformInfo,
)
from unieinfra_kernel.selection.registry import KernelRegistry, KernelSpec, Priority
from unieinfra_kernel.selection.signature import (
    dense_tensor_format,
    format_signature,
)

SIG = format_signature(
    a=dense_tensor_format(torch.bfloat16), b=dense_tensor_format(torch.bfloat16)
)

FAMILY, MODE = "autotunetest", "mm"


def _platform(device_name: str = "NVIDIA RTX PRO 6000") -> PlatformInfo:
    return PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(12, 0),
        device_name=device_name,
        device_count=1,
        total_memory=96 * 1024**3,
        memory_bandwidth=1792.0,
        sm_count=188,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=100 * 1024,
    )


@pytest.fixture
def two_kernels():
    """Register two interchangeable kernels for a throwaway operator."""
    registry = KernelRegistry.get()
    calls: dict[str, int] = {"fast": 0, "slow": 0, "boom": 0}

    def fast(**kwargs):
        calls["fast"] += 1
        return kwargs

    def slow(**kwargs):
        calls["slow"] += 1
        return kwargs

    def boom(**kwargs):
        calls["boom"] += 1
        raise RuntimeError("this kernel dies at runtime")

    specs = {
        # `slow` is the heuristic winner: higher band, and it carries the
        # `latency` tag so the objective dimension favours it too.
        "slow": KernelSpec(
            name="slow",
            family=FAMILY,
            mode=MODE,
            solution="vendorlib",
            format_signatures=frozenset({SIG}),
            priority=int(Priority.SPECIALIZED),
            tags=frozenset({"latency"}),
            capability=CapabilityRequirement(),
        ),
        "fast": KernelSpec(
            name="fast",
            family=FAMILY,
            mode=MODE,
            solution="triton",
            format_signatures=frozenset({SIG}),
            priority=int(Priority.PORTABLE),
            tags=frozenset({"throughput"}),
            origin="first_party",
            capability=CapabilityRequirement(),
        ),
        "boom": KernelSpec(
            name="boom",
            family=FAMILY,
            mode=MODE,
            solution="broken",
            format_signatures=frozenset({SIG}),
            priority=int(Priority.PLUGIN),
            capability=CapabilityRequirement(),
        ),
    }
    registry.register(specs["slow"], slow)
    registry.register(specs["fast"], fast)
    try:
        yield specs, calls
    finally:
        for name in ("slow", "fast", "boom"):
            registry._unregister(name)
        registry.clear_cache()
        sel.clear_kernel_pin()
        sel.set_selection_policy(sel.SelectionPolicy())
        sel.clear_autotune_log()


# ---------------------------------------------------------------------------
# Shape bucketing
# ---------------------------------------------------------------------------


def test_shape_bucket_rounds_m_up_to_power_of_two():
    assert shape_bucket({"M": 3, "N": 6144, "K": 4096})[0] == ("K", 4096)
    assert dict(shape_bucket({"M": 3, "N": 6144, "K": 4096}))["M"] == 4
    assert dict(shape_bucket({"M": 5, "N": 1, "K": 1}))["M"] == 8
    # 1, 2, 4 and 8 each keep their own bucket -- the crossover E33 measured
    # lives between them, so they must not merge.
    assert [dict(shape_bucket({"M": m}))["M"] for m in (1, 2, 4, 8)] == [1, 2, 4, 8]


def test_shape_bucket_keeps_n_and_k_exact_and_drops_labels():
    bucket = dict(shape_bucket({"M": 1, "N": 24576, "K": 4096, "label": "gate_up"}))
    assert bucket == {"M": 1, "N": 24576, "K": 4096}


def test_shape_bucket_of_nothing_is_empty():
    assert shape_bucket(None) == ()
    assert shape_bucket({}) == ()


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def test_cache_key_separates_devices_that_share_an_arch():
    """sm_120 spans the PRO 6000 and the 5090; they need not agree on a winner."""
    args = (FAMILY, MODE, SIG, {"m_le_8": True}, {"M": 1, "N": 4096, "K": 4096})
    pro = AutotuneCache.make_key(_platform("NVIDIA RTX PRO 6000"), *args)
    consumer = AutotuneCache.make_key(_platform("NVIDIA GeForce RTX 5090"), *args)
    assert pro != consumer
    # They agree on everything the arch-only key would have captured, so the
    # device name is the *only* thing keeping them apart.
    assert pro.replace("RTX PRO 6000", "") == consumer.replace("GeForce RTX 5090", "")


def test_cache_key_separates_shape_buckets_and_traits():
    base = AutotuneCache.make_key(
        _platform(), FAMILY, MODE, SIG, {"m_le_2": True}, {"M": 1, "N": 4096, "K": 4096}
    )
    other_m = AutotuneCache.make_key(
        _platform(), FAMILY, MODE, SIG, {"m_le_2": True}, {"M": 8, "N": 4096, "K": 4096}
    )
    other_traits = AutotuneCache.make_key(
        _platform(),
        FAMILY,
        MODE,
        SIG,
        {"m_le_2": False},
        {"M": 1, "N": 4096, "K": 4096},
    )
    assert len({base, other_m, other_traits}) == 3


def test_cache_roundtrips_through_disk(tmp_path):
    path = tmp_path / "cache.json"
    cache = AutotuneCache(path=path)
    cache.put("k", "fast", {"fast": 1.0, "slow": 2.0})
    cache.flush()
    assert path.is_file()

    reloaded = AutotuneCache(path=path)
    record = reloaded.get("k")
    assert record is not None
    assert record["kernel"] == "fast"
    assert record["times_us"]["slow"] == 2.0


def test_corrupt_cache_is_ignored_not_fatal(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json")
    assert AutotuneCache(path=path).get("anything") is None


# ---------------------------------------------------------------------------
# The selection path
# ---------------------------------------------------------------------------


def _autotune_gemm(monkeypatch, measurements, **kwargs):
    """Run one AUTOTUNE selection with measurement stubbed to `measurements`."""
    monkeypatch.setattr(sel, "autotune_available", lambda: True)
    monkeypatch.setattr(
        sel,
        "measure_candidates",
        lambda impls, *a, **kw: [
            Measurement(spec.name, **measurements[spec.name]) for spec, _ in impls
        ],
    )
    sel.enable_autotuning([(FAMILY, MODE)], params=AutotuneParams(use_cache=False))
    return sel.select_kernel(
        FAMILY,
        MODE,
        SIG,
        platform=_platform(),
        shape={"M": 1, "N": 4096, "K": 4096},
        **kwargs,
    )


def test_autotune_actually_measures_and_can_overturn_the_heuristic(
    monkeypatch, two_kernels
):
    """The whole point: the priority band loses to a stopwatch."""
    assert (
        sel.select_kernel(FAMILY, MODE, SIG, platform=_platform()).name == "slow"
    ), "precondition: heuristic prefers the higher-priority kernel"

    selected = _autotune_gemm(
        monkeypatch,
        {"slow": {"median_us": 100.0}, "fast": {"median_us": 40.0}},
    )
    assert selected.name == "fast"

    entry = sel.autotune_log()[-1]
    assert entry["heuristic"] == "slow"
    assert entry["measured"] == "fast"
    assert entry["disagreed"] is True


def test_autotune_keeps_the_heuristic_winner_inside_the_margin(
    monkeypatch, two_kernels
):
    """A 1% edge is noise; freezing a coin flip into the disk cache is worse."""
    selected = _autotune_gemm(
        monkeypatch,
        {"slow": {"median_us": 100.0}, "fast": {"median_us": 99.0}},
    )
    assert selected.name == "slow"
    assert sel.autotune_log()[-1]["disagreed"] is False


def test_a_candidate_that_raises_is_dropped_not_fatal(monkeypatch, two_kernels):
    """E30's hard-failing mis-gates: measurement should delete them silently."""
    selected = _autotune_gemm(
        monkeypatch,
        {
            "slow": {"error": "RuntimeError: no tcgen05 on this part"},
            "fast": {"median_us": 40.0},
        },
    )
    assert selected.name == "fast"
    assert sel.autotune_log()[-1]["dropped"] == ["slow"]


def test_every_candidate_failing_falls_back_to_the_heuristic(monkeypatch, two_kernels):
    selected = _autotune_gemm(
        monkeypatch,
        {"slow": {"error": "boom"}, "fast": {"error": "boom"}},
    )
    assert selected.name == "slow"


def test_autotune_no_longer_discards_the_callers_objective(monkeypatch, two_kernels):
    """The old code hardcoded DEFAULT, making AUTOTUNE worse than HEURISTIC."""
    captured: list = []
    monkeypatch.setattr(sel, "autotune_available", lambda: True)
    monkeypatch.setattr(
        sel,
        "measure_candidates",
        lambda impls, *a, **kw: []
        or (_ for _ in ()).throw(RuntimeError("no generator")),
    )
    real_rank = sel._rank_by_objective

    def spy(specs, objective, platform, traits):
        captured.append(objective)
        return real_rank(specs, objective, platform, traits)

    monkeypatch.setattr(sel, "_rank_by_objective", spy)
    sel.enable_autotuning([(FAMILY, MODE)], params=AutotuneParams(use_cache=False))
    sel.select_kernel(
        FAMILY,
        MODE,
        SIG,
        platform=_platform(),
        shape={"M": 1, "N": 4096, "K": 4096},
        objective=sel.SelectionObjective.THROUGHPUT,
    )
    assert sel.SelectionObjective.THROUGHPUT in captured
    assert sel.SelectionObjective.DEFAULT not in captured


def test_frozen_autotuning_never_measures(monkeypatch, two_kernels):
    """Tuning during serving blocks the launch thread and deadlocks TP."""
    monkeypatch.setattr(sel, "autotune_available", lambda: True)
    called = []
    monkeypatch.setattr(
        sel,
        "measure_candidates",
        lambda *a, **kw: called.append(1) or [Measurement("fast", median_us=1.0)],
    )
    sel.enable_autotuning([(FAMILY, MODE)], params=AutotuneParams(use_cache=False))
    with_freeze = sel.freeze_autotuning
    with_freeze()
    try:
        selected = sel.select_kernel(
            FAMILY,
            MODE,
            SIG,
            platform=_platform(),
            shape={"M": 1, "N": 4096, "K": 4096},
        )
    finally:
        from unieinfra_kernel.selection.tuning import unfreeze_autotuning

        unfreeze_autotuning()
    assert not called
    assert selected.name == "slow"


def test_a_single_candidate_is_never_measured(monkeypatch, two_kernels):
    """The M=256 finding, pinned.

    Selection can only fix what it can choose between. At ``M > 8`` every dense
    GEMM has exactly one candidate (``torch_linear``), so autotuning short-
    circuits — which is why E44's M=256 cuBLASLt loss is *not* reachable by
    better selection and needs a second kernel instead. This test exists so that
    conclusion cannot silently rot into "autotune measured it and agreed".
    """
    monkeypatch.setattr(sel, "autotune_available", lambda: True)
    called = []
    monkeypatch.setattr(sel, "measure_candidates", lambda *a, **kw: called.append(1))
    sel.enable_autotuning([(FAMILY, MODE)], params=AutotuneParams(use_cache=False))

    # Narrow the candidate set to one by pinning, the same way a trait gate does.
    with sel.pin_kernel(FAMILY, MODE, name="slow"):
        selected = sel.select_kernel(
            FAMILY,
            MODE,
            SIG,
            platform=_platform(),
            shape={"M": 256, "N": 6144, "K": 4096},
        )
    assert selected.name == "slow"
    assert not called, "one candidate means there is nothing to measure"


def test_no_shape_means_no_measurement(monkeypatch, two_kernels):
    monkeypatch.setattr(sel, "autotune_available", lambda: True)
    called = []
    monkeypatch.setattr(sel, "measure_candidates", lambda *a, **kw: called.append(1))
    sel.enable_autotuning([(FAMILY, MODE)], params=AutotuneParams(use_cache=False))
    assert sel.select_kernel(FAMILY, MODE, SIG, platform=_platform()).name == "slow"
    assert not called


def test_second_lookup_hits_the_disk_cache_without_re_measuring(
    monkeypatch, tmp_path, two_kernels
):
    """This is what makes the second engine startup fast."""
    monkeypatch.setenv("UNIEINFRA_KERNEL_AUTOTUNE_CACHE", str(tmp_path))
    monkeypatch.setattr(sel, "autotune_available", lambda: True)
    measured = []

    def measure(impls, *a, **kw):
        measured.append(1)
        return [
            Measurement(spec.name, median_us=(40.0 if spec.name == "fast" else 100.0))
            for spec, _ in impls
        ]

    monkeypatch.setattr(sel, "measure_candidates", measure)
    sel.enable_autotuning([(FAMILY, MODE)], params=AutotuneParams())

    first = sel.select_kernel(
        FAMILY, MODE, SIG, platform=_platform(), shape={"M": 1, "N": 4096, "K": 4096}
    )
    assert first.name == "fast" and len(measured) == 1

    # Drop the in-memory memo and the process-wide cache handle: this is the
    # "second startup" case, not a repeated call in one process.
    KernelRegistry.get().clear_cache()
    from unieinfra_kernel.selection import autotune as autotune_mod

    autotune_mod._cache_by_path.clear()

    second = sel.select_kernel(
        FAMILY, MODE, SIG, platform=_platform(), shape={"M": 1, "N": 4096, "K": 4096}
    )
    assert second.name == "fast"
    assert len(measured) == 1, "the cached decision must not be re-measured"
    assert sel.autotune_log()[-1]["source"] == "disk-cache"


# ---------------------------------------------------------------------------
# Pins (benchmark ablations)
# ---------------------------------------------------------------------------


def test_pin_by_name_restricts_selection(two_kernels):
    with sel.pin_kernel(FAMILY, MODE, name="fast"):
        assert sel.select_kernel(FAMILY, MODE, SIG, platform=_platform()).name == "fast"
    assert sel.select_kernel(FAMILY, MODE, SIG, platform=_platform()).name == "slow"


def test_pin_by_origin_restricts_selection(two_kernels):
    with sel.pin_kernel(origin="first_party"):
        assert sel.select_kernel(FAMILY, MODE, SIG, platform=_platform()).name == "fast"
    with sel.pin_kernel(origin="vendor"):
        assert sel.select_kernel(FAMILY, MODE, SIG, platform=_platform()).name == "slow"


def test_pin_by_solution_restricts_selection(two_kernels):
    with sel.pin_kernel(FAMILY, MODE, solution="triton"):
        assert sel.select_kernel(FAMILY, MODE, SIG, platform=_platform()).name == "fast"


def test_pin_with_no_match_raises_rather_than_silently_ignoring(two_kernels):
    with sel.pin_kernel(origin="kda"):
        with pytest.raises(sel.NoKernelFoundError):
            sel.select_kernel(FAMILY, MODE, SIG, platform=_platform())


def test_env_pin_is_honoured(monkeypatch, two_kernels):
    monkeypatch.setenv("UNIEINFRA_KERNEL_PIN", "origin=first_party")
    KernelRegistry.get().clear_cache()
    assert sel.select_kernel(FAMILY, MODE, SIG, platform=_platform()).name == "fast"


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def test_warmup_accepts_four_and_five_tuples_and_reports_seconds(two_kernels):
    seconds = sel.warmup_selection(
        [
            (FAMILY, MODE, SIG, None),
            (FAMILY, MODE, SIG, None, {"M": 1, "N": 4096, "K": 4096}),
        ]
    )
    assert seconds >= 0.0


def test_warmup_can_freeze_when_it_finishes(two_kernels):
    from unieinfra_kernel.selection.tuning import (
        autotune_frozen,
        unfreeze_autotuning,
    )

    try:
        sel.warmup_selection([(FAMILY, MODE, SIG, None)], freeze=True)
        assert autotune_frozen()
    finally:
        unfreeze_autotuning()


# ---------------------------------------------------------------------------
# Real measurement (GPU)
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@requires_cuda
def test_gemm_input_generator_builds_runnable_dense_inputs():
    from unieinfra_kernel.contracts.gemm import generate_gemm_inputs
    from unieinfra_kernel.ops.gemm.torch_linear import torch_linear

    inputs = generate_gemm_inputs(
        format_signature=SIG, shape={"M": 2, "N": 512, "K": 256}, device="cuda"
    )
    out = torch_linear(**inputs)
    assert out.shape == (2, 512)
    assert out.dtype == torch.bfloat16


@requires_cuda
def test_measure_candidates_drops_a_kernel_that_raises():
    from unieinfra_kernel.ops.gemm.torch_linear import torch_linear
    from unieinfra_kernel.selection.autotune import measure_candidates

    good = KernelSpec(name="good", family="gemm", mode="mm")
    bad = KernelSpec(name="bad", family="gemm", mode="mm")

    def explodes(**kwargs):
        raise RuntimeError("simulated illegal configuration")

    results = measure_candidates(
        [(good, torch_linear), (bad, explodes)],
        "gemm",
        "mm",
        SIG,
        {"M": 1, "N": 512, "K": 256},
        AutotuneParams(warmup_iters=1, bench_iters=3, reps=1),
    )
    by_name = {m.name: m for m in results}
    assert by_name["good"].ok
    assert not by_name["bad"].ok
    assert "simulated illegal configuration" in by_name["bad"].error
