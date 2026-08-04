# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Proof that a dropped-in KDA kernel is discovered, registered, and selectable.

Runs on CPU: selection is pure Python and platforms are synthesized, so no GPU
is touched. See ``conftest.py`` for how the ``tokenspeed_kernel`` core is made
importable without the GPU toolchain.

The suite proves the three claims the scaffold rests on:

1. Importing ``unieinfra_kernel`` self-registers the KDA kernel into the *same*
   ``tokenspeed_kernel`` registry (no second registry).
2. On the matching platform (sm_120) the KDA kernel wins a same-priority tie
   against a vendor kernel via the ``origin`` tiebreaker; and it does so under
   both the HEURISTIC and AUTOTUNE strategies.
3. The ``origin`` axis never disturbs the existing (all-``vendor``) corpus: a
   pre-sm_120 platform still picks the vendor kernel, two equal-priority vendor
   kernels keep their registration-order winner, and unrelated ops are
   unchanged after discovery.
"""

from __future__ import annotations

import torch

# Importing the package triggers discovery -> the KDA kernel self-registers.
import unieinfra_kernel  # noqa: F401
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    PlatformInfo,
    _get_cuda_sm_features,
)
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.selection import (
    SelectionPolicy,
    SelectionStrategy,
    _score_origin,
    select_kernel,
    set_selection_policy,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from unieinfra_kernel.contracts.activation import SILU_AND_MUL

_SIG = format_signature(x=dense_tensor_format(torch.bfloat16))


def _platform(major: int, minor: int) -> PlatformInfo:
    """Synthesize a minimal NVIDIA platform at the given arch (no GPU needed)."""
    return PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(major, minor),
        device_name=f"synthetic-sm_{major}{minor}",
        device_count=1,
        total_memory=0,
        memory_bandwidth=0.0,
        sm_count=0,
        max_threads_per_sm=0,
        max_shared_memory_per_sm=0,
        # Derived from the real detector rather than hand-copied, so the
        # feature set under test can never drift from the one the platform
        # layer actually produces (E30 fixed three fixtures that had).
        sm_features=_get_cuda_sm_features(ArchVersion(major, minor)),
    )


SM120 = _platform(12, 0)  # RTX PRO 6000 (Blackwell workstation)
SM90 = _platform(9, 0)  # Hopper — pre-sm_120


def _register_vendor_silu(name: str) -> None:
    """Register a vendor incumbent for activation.silu_and_mul.

    Broad capability (sm_80+) and the same priority band as the KDA kernel, so on
    sm_120 the two tie on (oracle, objective, priority) and only ``origin``
    separates them.
    """

    @register_kernel(
        SILU_AND_MUL.family,
        SILU_AND_MUL.mode,
        name=name,
        solution="vendor_ref",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(8, 0),
            required_features=frozenset({"tensor_core:f16"}),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=SILU_AND_MUL.signatures,
        traits=SILU_AND_MUL.traits,
        priority=Priority.PERFORMANT + 2,  # identical to the KDA kernel
        origin="vendor",
    )
    def _vendor_silu(x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


def test_kda_kernel_is_discovered_and_registered() -> None:
    reg = KernelRegistry.get()
    spec = reg.get_by_name("kda_silu_and_mul_sm120")
    assert spec is not None, "discovery did not register the KDA kernel"
    assert spec.origin == "kda"
    assert (spec.family, spec.mode) == SILU_AND_MUL.operator
    # Registered into the SAME shared registry the runtime consumes.
    assert reg.get_impl("kda_silu_and_mul_sm120") is not None


def test_kda_wins_same_priority_tie_via_origin_on_sm120() -> None:
    _register_vendor_silu("vendor_silu_tie")

    selected = select_kernel(
        SILU_AND_MUL.family, SILU_AND_MUL.mode, _SIG, platform=SM120
    )
    assert (
        selected.name == "kda_silu_and_mul_sm120"
    ), "KDA kernel should win the same-priority tie on sm_120 via origin"

    # It is genuinely runnable (correctness against the contract reference).
    x = torch.randn(4, 2048, dtype=torch.float32)
    out = selected(x)
    torch.testing.assert_close(out, SILU_AND_MUL.reference(x))


def test_autotune_strategy_also_honors_origin_tiebreak() -> None:
    _register_vendor_silu("vendor_silu_autotune")
    set_selection_policy(
        SelectionPolicy(
            op_strategies={
                SILU_AND_MUL.operator: SelectionStrategy.AUTOTUNE,
            }
        )
    )
    try:
        selected = select_kernel(
            SILU_AND_MUL.family, SILU_AND_MUL.mode, _SIG, platform=SM120
        )
        assert selected.name == "kda_silu_and_mul_sm120"
    finally:
        set_selection_policy(SelectionPolicy())  # restore default HEURISTIC


def test_kda_does_not_displace_vendor_on_pre_sm120() -> None:
    _register_vendor_silu("vendor_silu_presm120")
    # On Hopper the KDA kernel is filtered out by its sm_120 capability, so the
    # vendor incumbent still wins even though they share a priority band.
    selected = select_kernel(
        SILU_AND_MUL.family, SILU_AND_MUL.mode, _SIG, platform=SM90
    )
    assert selected.name.startswith("vendor_silu"), (
        "a same-priority KDA kernel must not displace vendor where it is not "
        "specialized"
    )


def test_origin_never_reorders_two_equal_priority_vendor_kernels() -> None:
    """The invariant: among all-vendor kernels, origin is constant -> no reorder.

    Two vendor kernels at identical priority keep the registration-order winner,
    exactly as before the origin axis existed (origin rank is 0 for both, so the
    appended sort dimension is a no-op and the stable sort is unchanged).
    """
    family, mode = "unieinfra_demo", "eq_prio"

    @register_kernel(
        family,
        mode,
        name="vendor_eq_first",
        solution="ref",
        signatures={_SIG},
        priority=Priority.PERFORMANT,
        origin="vendor",
    )
    def _first(x: torch.Tensor) -> torch.Tensor:
        return x

    @register_kernel(
        family,
        mode,
        name="vendor_eq_second",
        solution="ref",
        signatures={_SIG},
        priority=Priority.PERFORMANT,
        origin="vendor",
    )
    def _second(x: torch.Tensor) -> torch.Tensor:
        return x

    first = KernelRegistry.get().get_by_name("vendor_eq_first")
    second = KernelRegistry.get().get_by_name("vendor_eq_second")
    assert _score_origin(first) == _score_origin(second) == 0

    selected = select_kernel(family, mode, _SIG, platform=SM120)
    assert (
        selected.name == "vendor_eq_first"
    ), "equal-priority vendor kernels must keep the registration-order winner"


def test_priority_still_dominates_origin() -> None:
    """Origin is only a tiebreaker: a higher-priority vendor beats a kda kernel."""
    family, mode = "unieinfra_demo", "prio_dominates"

    @register_kernel(
        family,
        mode,
        name="kda_low_prio",
        solution="ref",
        signatures={_SIG},
        priority=Priority.PORTABLE,
        origin="kda",
    )
    def _kda(x: torch.Tensor) -> torch.Tensor:
        return x

    @register_kernel(
        family,
        mode,
        name="vendor_high_prio",
        solution="ref",
        signatures={_SIG},
        priority=Priority.SPECIALIZED,
        origin="vendor",
    )
    def _vendor(x: torch.Tensor) -> torch.Tensor:
        return x

    selected = select_kernel(family, mode, _SIG, platform=SM120)
    assert (
        selected.name == "vendor_high_prio"
    ), "priority must dominate; origin only breaks exact ties"
