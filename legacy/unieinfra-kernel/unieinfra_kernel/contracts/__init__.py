# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Frozen KernelContracts, one per (family, mode).

A :class:`KernelContract` is simultaneously the registry's source of truth for
an op *and* a kernel-design-agent's task spec (see
``docs/design/kernel-plugin-contract.md``). It bundles exactly the fields the
contest / KDA per-kernel spec uses:

    signature · traits axes · reference · tolerance · workloads · capability

Contracts live under ``contracts/`` and are imported explicitly by kernels and
tests; the discovery walker does **not** scan this tree (only ``ops/``, ``kda/``,
``contrib/``), so a contract module never accidentally self-registers a kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from unieinfra_kernel.boundary import CapabilityRequirement, FormatSignature

__all__ = ["Tolerance", "KernelContract"]


@dataclass(frozen=True)
class Tolerance:
    """Correctness-gate tolerance, mirroring the KDA ``--atol/--rtol/--ratio`` args."""

    atol: float = 1e-2
    rtol: float = 1e-2
    # Fraction of elements that must be within tolerance (KDA "matched ratio").
    required_matched_ratio: float = 1.0


@dataclass(frozen=True)
class KernelContract:
    """The frozen contract for one operator ``(family, mode)``.

    This is the object an agent targets and the registry treats as source of
    truth. A kernel is drop-in *by construction* when it declares the contract's
    ``family``/``mode``/``signatures``/``capability`` on ``@register_kernel`` and
    passes the correctness gate against ``reference`` at ``tolerance`` over
    ``workloads``.

    Attributes:
        family: Operator family, e.g. ``"activation"``.
        mode: Operator mode, e.g. ``"silu_and_mul"``.
        signatures: Accepted role-indexed tensor format signatures.
        traits: Trait axes the op is defined over (constant/variable axes in KDA
            terms), each mapping a trait name to its allowed values.
        reference: Correctness reference implementation ``impl(*args) -> out``.
        tolerance: Acceptance tolerance for the correctness gate.
        workloads: Representative shapes for autotune / benchmarking. Each entry
            is a plain dict of shape parameters understood by the op's harness.
        capability: Platform capability the contract targets.
    """

    family: str
    mode: str
    signatures: frozenset[FormatSignature]
    reference: Callable[..., Any]
    tolerance: Tolerance = field(default_factory=Tolerance)
    traits: dict[str, frozenset[Any]] = field(default_factory=dict)
    workloads: tuple[dict[str, Any], ...] = ()
    capability: CapabilityRequirement = field(default_factory=CapabilityRequirement)

    @property
    def operator(self) -> tuple[str, str]:
        """The ``(family, mode)`` key this contract governs."""
        return (self.family, self.mode)
