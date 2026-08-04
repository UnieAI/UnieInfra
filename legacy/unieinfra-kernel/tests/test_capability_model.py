# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""The capability model itself: the mis-gating bug class must be unwritable.

E30 (our tree), E28 (our tree) and E37 (upstream vLLM, now vendored) are three
measured instances of one bug: an **open-ended architecture comparison** used as
a stand-in for a capability. This suite pins the model that makes that form
impossible to write rather than merely discouraged.

Every test here runs on CPU -- platforms are synthesized through the real
detector, so no GPU is touched and no fixture can drift from the code under
test.

Each test is written to FAIL if the corresponding fix is reverted:

* the ``pytest.raises`` tests fail (nothing raised) without the ``__post_init__``
  validation;
* the ``compute:cluster`` tests fail without the driver-sourced feature, because
  the pure ``major >= 9`` inference cannot report an H20 or a cluster-less part;
* the ``build:`` tests fail with ``AttributeError``/``ImportError`` without the
  artifact tier;
* ``test_no_registered_kernel_carries_an_open_ended_gate`` fails against the
  pre-E40 corpus, which had 25 such registrations.
"""

from __future__ import annotations

import pytest
from unieinfra_kernel.selection.artifact import (
    _fatbin_covers,
    build_capability_available,
    clear_build_capability_cache,
    has_signature_parameter,
    kernel_is_registered,
    register_build_capability,
    symbol_is_bound,
)
from unieinfra_kernel.selection.platform import (
    ArchVersion,
    CapabilityRequirement,
    InterconnectInfo,
    PlatformInfo,
    _get_cuda_sm_features,
)

# ---------------------------------------------------------------------------
# Synthetic platforms, derived from the real detector (never hand-copied).
# ---------------------------------------------------------------------------


def _nvidia(
    major: int,
    minor: int,
    name: str,
    *,
    cluster_launch: bool | None = None,
) -> PlatformInfo:
    arch_version = ArchVersion(major, minor)
    return PlatformInfo(
        vendor="nvidia",
        arch_version=arch_version,
        device_name=name,
        device_count=1,
        total_memory=80 * (1024**3),
        memory_bandwidth=3350.0,
        sm_count=132,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=232448,
        sm_features=_get_cuda_sm_features(
            arch_version, cluster_launch=cluster_launch, device_name=name
        ),
        runtime_features=frozenset({"runtime:cuda_graph"}),
        interconnect=InterconnectInfo(topology="single_gpu"),
    )


SM86 = _nvidia(8, 6, "NVIDIA RTX A6000")
SM89 = _nvidia(8, 9, "NVIDIA L40S")
SM90 = _nvidia(9, 0, "NVIDIA H200")
SM100 = _nvidia(10, 0, "NVIDIA B200")
SM103 = _nvidia(10, 3, "NVIDIA B300")
SM120 = _nvidia(12, 0, "NVIDIA RTX PRO 6000 Blackwell")

DATACENTER = {"sm_90": SM90, "sm_100": SM100, "sm_103": SM103}


# ---------------------------------------------------------------------------
# Rule 1: an open-ended min_arch_version is unrepresentable
# ---------------------------------------------------------------------------


def test_min_arch_alone_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError) as excinfo:
        CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        )
    message = str(excinfo.value)
    # The message must say what to ADD and why -- an author hitting this needs
    # the remedy, not just a rejection.
    assert "required_features" in message
    assert "max_arch_version" in message
    assert "sm_120" in message


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "min_arch_version": ArchVersion(10, 0),
                "max_arch_version": ArchVersion(10, 9),
            },
            id="explicit-upper-bound",
        ),
        pytest.param(
            {
                "min_arch_version": ArchVersion(10, 0),
                "required_features": frozenset({"compute:tcgen05"}),
            },
            id="capability-requirement",
        ),
    ],
)
def test_either_remedy_is_accepted(kwargs: dict) -> None:
    requirement = CapabilityRequirement(vendors=frozenset({"nvidia"}), **kwargs)
    # Both remedies exclude sm_120 and leave datacenter Blackwell untouched.
    assert requirement.satisfied_by(SM120) is False
    assert requirement.satisfied_by(SM100) is True
    assert requirement.satisfied_by(SM103) is True


def test_max_arch_alone_is_allowed() -> None:
    """An upper bound with no lower bound is not the bug shape."""
    requirement = CapabilityRequirement(
        max_arch_version=ArchVersion(9, 9),
        vendors=frozenset({"nvidia"}),
    )
    assert requirement.satisfied_by(SM90) is True
    assert requirement.satisfied_by(SM100) is False


def test_features_only_requirement_is_allowed_and_vendor_neutral() -> None:
    """A pure capability statement needs no arch bound and no vendor."""
    requirement = CapabilityRequirement(required_features=frozenset({"tensor_core:f4"}))
    assert requirement.satisfied_by(SM100) is True
    assert requirement.satisfied_by(SM120) is True
    assert requirement.satisfied_by(SM90) is False


def test_empty_requirement_is_allowed() -> None:
    """The default (portable, everywhere) requirement must stay constructible."""
    assert CapabilityRequirement().satisfied_by(SM86) is True


# ---------------------------------------------------------------------------
# Rule 4: vendor must be explicit whenever an arch bound is present
# ---------------------------------------------------------------------------


def test_arch_bound_without_vendor_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        )
    message = str(excinfo.value)
    assert "vendors" in message
    # The reason must be stated: the numbers are shared between vendors.
    assert "gfx950" in message or "CDNA" in message


def test_the_collision_that_makes_rule_4_necessary() -> None:
    """AMD gfx950 is ArchVersion(9, 5) -- the same number space NVIDIA uses.

    Without ``vendors``, a Hopper-era gate silently admits CDNA4, and a sm_12x
    gate silently admits CDNA5 (12, 5). With it, they cannot.
    """
    amd_cdna4 = PlatformInfo(
        vendor="amd",
        arch_version=ArchVersion(9, 5),
        device_name="AMD Instinct MI350X",
        device_count=1,
        total_memory=0,
        memory_bandwidth=0.0,
        sm_count=256,
        max_threads_per_sm=0,
        max_shared_memory_per_sm=160 * 1024,
        sm_features=frozenset({"tensor_core:f16", "tensor_core:f8", "tensor_core:f4"}),
    )
    nvidia_only = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        required_features=frozenset({"memory:tma"}),
        vendors=frozenset({"nvidia"}),
    )
    assert nvidia_only.satisfied_by(amd_cdna4) is False
    assert nvidia_only.satisfied_by(SM90) is True


def test_unknown_and_empty_vendors_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown vendor"):
        CapabilityRequirement(vendors=frozenset({"intel"}))
    with pytest.raises(ValueError, match="never be satisfied"):
        CapabilityRequirement(vendors=frozenset())


def test_inverted_arch_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="below min_arch_version"):
        CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        )


# ---------------------------------------------------------------------------
# Rule 2: capabilities come from measured device properties
# ---------------------------------------------------------------------------


def test_cluster_launch_comes_from_the_driver_not_the_version() -> None:
    """The driver answer wins over the arch inference, in both directions."""
    # An sm_90-class part the driver says cannot launch clusters (H20).
    h20 = _nvidia(9, 0, "NVIDIA H20", cluster_launch=False)
    assert h20.has_cluster_launch is False
    assert "compute:cluster" not in h20.sm_features
    # ... while it keeps every other Hopper capability, so this narrows exactly
    # one feature and not the whole generation.
    assert "memory:tma" in h20.sm_features

    # sm_120 genuinely HAS clusters (ptxas + driver, E30) -- an inference that
    # tied clusters to "datacenter" would get this wrong in the other direction.
    sm120 = _nvidia(12, 0, "NVIDIA RTX PRO 6000 Blackwell", cluster_launch=True)
    assert sm120.has_cluster_launch is True


def test_cluster_launch_falls_back_to_validated_behaviour_when_undetectable() -> None:
    """No driver answer must never narrow what already worked (the E28 rule)."""
    for platform in DATACENTER.values():
        # Built with cluster_launch=None, i.e. the driver could not be asked.
        assert platform.has_cluster_launch is True

    # And the pre-driver-query H20 workaround is preserved on that same path,
    # without an H200 being caught by it.
    h20 = _nvidia(9, 0, "NVIDIA H20")
    h200 = _nvidia(9, 0, "NVIDIA H200")
    assert h20.has_cluster_launch is False
    assert h200.has_cluster_launch is True


def test_datacenter_feature_sets_are_unchanged() -> None:
    """Byte-identical on sm_90 / sm_100 / sm_103 -- the hardware we cannot test.

    Spelled out as literals on purpose: this is the assertion that a future
    edit to the arch -> feature table cannot silently move datacenter parts.
    ``tensor_core:block_scale`` is the one addition, and it is purely additive
    (it names a capability those parts always had).
    """
    assert SM90.sm_features == frozenset(
        {
            "tensor_core:f16",
            "tensor_core:int8",
            "tensor_core:f8",
            "memory:async_copy",
            "memory:tma",
            "compute:cluster",
        }
    )
    blackwell = frozenset(
        {
            "tensor_core:f16",
            "tensor_core:int8",
            "tensor_core:f8",
            "tensor_core:f4",
            "tensor_core:block_scale",
            "memory:async_copy",
            "memory:tma",
            "compute:cluster",
            "compute:tcgen05",
        }
    )
    assert SM100.sm_features == blackwell
    assert SM103.sm_features == blackwell


def test_sm120_is_not_datacenter_blackwell() -> None:
    """The distinction the whole model exists to draw."""
    assert SM120.has_fp4_tensor_cores is True  # it does have FP4 tensor cores
    assert SM120.has_cluster_launch is True  # and clusters + distributed smem
    assert SM120.has_tcgen05 is False  # but no tcgen05 / TMEM
    assert "tensor_core:block_scale" in SM120.sm_features


# ---------------------------------------------------------------------------
# Rule 5: a part we have never seen must be classified correctly by
# construction -- no code change required.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "major,minor,expect_tcgen05,expect_fp4",
    [
        # B300 is already newer than the sm_100 most gates were written for.
        pytest.param(10, 3, True, True, id="sm_103-b300"),
        # A hypothetical datacenter Blackwell refresh.
        pytest.param(10, 9, True, True, id="sm_109-unseen-datacenter"),
        # A hypothetical workstation part in the sm_12x family.
        pytest.param(12, 1, False, True, id="sm_121-unseen-workstation"),
        # A whole generation beyond anything in the table: it must NOT inherit
        # tcgen05, because tcgen05 is bounded, not open-ended.
        pytest.param(13, 0, False, True, id="sm_130-unseen-generation"),
    ],
)
def test_unseen_parts_are_classified_without_a_code_change(
    major: int, minor: int, expect_tcgen05: bool, expect_fp4: bool
) -> None:
    platform = _nvidia(major, minor, f"NVIDIA UNSEEN sm_{major}{minor}")
    assert platform.has_tcgen05 is expect_tcgen05
    assert platform.has_fp4_tensor_cores is expect_fp4

    tcgen05_kernel = CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        required_features=frozenset({"compute:tcgen05"}),
        vendors=frozenset({"nvidia"}),
    )
    assert tcgen05_kernel.satisfied_by(platform) is expect_tcgen05


# ---------------------------------------------------------------------------
# Rule 3: the build/artifact capability tier
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_build_cache():
    clear_build_capability_cache()
    yield
    clear_build_capability_cache()


def test_build_capability_gates_a_requirement() -> None:
    name = "build:_test_present"
    register_build_capability(name, lambda: True, default=False)
    requirement = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        required_features=frozenset({name}),
        vendors=frozenset({"nvidia"}),
    )
    assert requirement.satisfied_by(SM100) is True

    register_build_capability(name, lambda: False, default=True)
    assert requirement.satisfied_by(SM100) is False
    assert requirement.missing_features(SM100) == {name}


def test_failing_probe_falls_back_to_its_declared_default() -> None:
    """A probe that cannot run must not decide the answer by accident."""

    def _boom() -> bool:
        raise RuntimeError("toolchain missing")

    register_build_capability("build:_test_boom_open", _boom, default=True)
    register_build_capability("build:_test_boom_closed", _boom, default=False)
    assert build_capability_available("build:_test_boom_open") is True
    assert build_capability_available("build:_test_boom_closed") is False


def test_unprobed_build_feature_never_narrows() -> None:
    """The E28 rule, encoded: unknown must not mean unavailable.

    A ``build:`` feature nobody registered a probe for resolves to AVAILABLE, so
    a detection gap can never delete a kernel that is validated on hardware this
    process cannot query.
    """
    requirement = CapabilityRequirement(
        required_features=frozenset({"build:_test_never_registered"})
    )
    assert requirement.satisfied_by(SM100) is True


def test_build_namespace_is_enforced() -> None:
    with pytest.raises(ValueError, match="must start with"):
        register_build_capability("not_a_build_feature", lambda: True, default=True)


def test_probe_constructors_cover_the_three_in_tree_patterns() -> None:
    """The three hand-rolled shapes, now one mechanism."""

    # 1. inspect.signature probe.
    def _with_sfq(q, k, v, sfq=None):
        return None

    def _without_sfq(q, k, v):
        return None

    assert has_signature_parameter(lambda: _with_sfq, "sfq")() is True
    assert has_signature_parameter(lambda: _without_sfq, "sfq")() is False

    # 2. try/except + `is error_fn` sentinel probe.
    sentinel = object()

    def _boom():
        raise ImportError("optional dependency absent")

    assert symbol_is_bound(lambda: _with_sfq, sentinels=(sentinel,))() is True
    assert symbol_is_bound(lambda: sentinel, sentinels=(sentinel,))() is False
    assert symbol_is_bound(lambda: None)() is False
    assert symbol_is_bound(_boom)() is False

    # 3. registry query probe.
    assert kernel_is_registered("no_such_kernel_anywhere")() is False


@pytest.mark.parametrize(
    "listing,target,expected",
    [
        # The real answer for this tree's prebuilt merge_state.so, measured with
        # cuobjdump: SASS for sm_120a only. A datacenter part is NOT covered --
        # which is exactly the question `min_arch_version=(9, 0)` could not ask.
        pytest.param(
            "ELF file    1: merge_state.1.sm_120a.cubin", 120, True, id="sm120-elf-hit"
        ),
        pytest.param(
            "ELF file    1: merge_state.1.sm_120a.cubin",
            100,
            False,
            id="sm100-elf-miss",
        ),
        pytest.param(
            "ELF file    1: k.1.sm_100a.cubin", 103, False, id="sass-is-exact-match"
        ),
        # PTX JITs forward, but only inside the same major family.
        pytest.param(
            "PTX file    1: k.1.sm_100.ptx", 103, True, id="ptx-jits-forward-in-family"
        ),
        pytest.param(
            "PTX file    1: k.1.sm_90.ptx", 100, False, id="ptx-does-not-cross-family"
        ),
        pytest.param("", 100, False, id="empty-listing"),
    ],
)
def test_fatbin_coverage(listing: str, target: int, expected: bool) -> None:
    assert _fatbin_covers(listing, target) is expected


# ---------------------------------------------------------------------------
# Corpus-level guard
# ---------------------------------------------------------------------------


def test_no_registered_kernel_carries_an_open_ended_gate() -> None:
    """Nothing in the live registry may carry the mis-gating form.

    Construction now rejects it, so this can only fail if some path builds a
    ``CapabilityRequirement`` around the validation -- which is precisely what a
    regression would look like.
    """
    import unieinfra_kernel  # noqa: F401  (import triggers discovery)
    from unieinfra_kernel.boundary import KernelRegistry

    offenders = [
        spec.name
        for spec in KernelRegistry.get().list_kernels()
        if spec.capability.min_arch_version is not None
        and spec.capability.max_arch_version is None
        and not spec.capability.required_features
    ]
    assert offenders == []


def test_no_registered_kernel_has_an_implicit_vendor() -> None:
    import unieinfra_kernel  # noqa: F401  (import triggers discovery)
    from unieinfra_kernel.boundary import KernelRegistry

    offenders = [
        spec.name
        for spec in KernelRegistry.get().list_kernels()
        if spec.capability.vendors is None
        and (
            spec.capability.min_arch_version is not None
            or spec.capability.max_arch_version is not None
        )
    ]
    assert offenders == []
