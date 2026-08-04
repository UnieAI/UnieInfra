# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Concrete KernelContract for the ``activation.silu_and_mul`` operator.

This is the worked example that realizes ``docs/design/kernel-plugin-contract.md``:
a single, frozen contract that a KDA agent can target and that the demo KDA
kernel in ``unieinfra_kernel/kda/activation/silu_and_mul_sm120.py`` fulfils.

The op is the SwiGLU/GeGLU-style gated activation: given ``x`` of shape
``(..., 2 * d)``, return ``silu(x[..., :d]) * x[..., d:]``. It is chosen because
it is genuinely used in transformer MLPs yet is trivially correct on CPU, so the
contract's reference doubles as a runnable correctness oracle in tests.
"""

from __future__ import annotations

import torch
from unieinfra_kernel.boundary import (
    ArchVersion,
    CapabilityRequirement,
    dense_tensor_format,
    format_signature,
)
from unieinfra_kernel.contracts import KernelContract, Tolerance

__all__ = ["FAMILY", "MODE", "SILU_AND_MUL", "silu_and_mul_reference"]

FAMILY = "activation"
MODE = "silu_and_mul"

# One tensor role ("x"), dense bf16/fp16. These are the exact signatures a kernel
# must declare to satisfy the contract and that selection matches against.
_SIGNATURES = frozenset(
    {
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        format_signature(x=dense_tensor_format(torch.float16)),
    }
)


def silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU gate: ``silu(x[..., :d]) * x[..., d:]``.

    Args:
        x: Input tensor whose last dim is even, ``(..., 2 * d)``.

    Returns:
        Tensor of shape ``(..., d)`` in the same dtype as ``x``.
    """
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


# sm_120 (RTX PRO 6000 Blackwell workstation) is the platform this contract is
# published for — the datacenter flashinfer/tcgen05 winners do not cover it, so
# it is exactly where KDA-generated kernels earn their place.
#
# The bound is explicit on BOTH ends. Without ``max_arch_version`` this would be
# an open-ended ``>=`` that silently claims every future NVIDIA part, which is
# the mis-gating shape E30 catalogued; the contract's tuned launch configuration
# is qualified on the sm_12x family and nothing else, so a newer part must be
# re-qualified rather than assumed. ``vendors`` is equally load-bearing here:
# AMD CDNA5 is ``ArchVersion(12, 5)`` and would otherwise fall inside this range.
SM120 = CapabilityRequirement(
    min_arch_version=ArchVersion(12, 0),
    max_arch_version=ArchVersion(12, 9),
    vendors=frozenset({"nvidia"}),
)

SILU_AND_MUL = KernelContract(
    family=FAMILY,
    mode=MODE,
    signatures=_SIGNATURES,
    reference=silu_and_mul_reference,
    tolerance=Tolerance(atol=1e-2, rtol=1e-2, required_matched_ratio=0.99),
    traits={"gated": frozenset({True})},
    workloads=(
        {"num_tokens": 128, "d": 1024},
        {"num_tokens": 4096, "d": 4096},
        {"num_tokens": 16384, "d": 8192},
    ),
    capability=SM120,
)
