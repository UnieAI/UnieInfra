# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""KDA-generated ``activation.silu_and_mul`` kernel, gated to sm_120.

This is the worked drop-in example for the plugin contract. It fulfils
:data:`unieinfra_kernel.contracts.activation.SILU_AND_MUL` and is registered with
``origin="kda"``. It demonstrates the whole promise of the scaffold:

    drop a file under ``kda/<family>/`` -> discovery imports it -> it self-registers
    -> ``select_kernel`` picks it on the matching platform, with **zero** edits to
    the registry or selection core.

Selection semantics demonstrated:

* It declares the **same** ``(family, mode)``, signatures, and priority as the
  vendor baseline, and is capability-gated to sm_120. On an sm_120 platform it
  therefore ties the vendor kernel on ``(oracle, objective, priority)`` and wins
  on the ``origin`` tiebreaker (``kda`` outranks ``vendor``).
* On any pre-sm_120 platform it is filtered out by capability, so the vendor
  kernel still wins — a same-priority KDA kernel never displaces the vendor
  default where it is not actually specialized.

The body here is a portable Torch reference so the example is runnable and
correct on CPU today; a real KDA submission would replace it with a CuteDSL
(NVIDIA) / Gluon (AMD) / Triton body behind the same registration.
"""

from __future__ import annotations

import torch
from unieinfra_kernel.boundary import Priority, register_kernel
from unieinfra_kernel.contracts.activation import (
    SILU_AND_MUL,
    silu_and_mul_reference,
)

__all__ = ["kda_silu_and_mul_sm120"]


@register_kernel(
    SILU_AND_MUL.family,
    SILU_AND_MUL.mode,
    name="kda_silu_and_mul_sm120",
    solution="kda_torch",  # placeholder backend; real submission swaps the body
    capability=SILU_AND_MUL.capability,
    signatures=SILU_AND_MUL.signatures,
    traits=SILU_AND_MUL.traits,
    # Same band as the in-tree default so the win comes from `origin`, not from
    # inflating priority. A hardened KDA kernel could instead claim the PLUGIN
    # band to win outright; here we intentionally exercise the tiebreaker.
    priority=Priority.PERFORMANT + 2,
    origin="kda",
    tags={"latency"},
)
def kda_silu_and_mul_sm120(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU gate ``silu(x[..., :d]) * x[..., d:]`` (sm_120, KDA).

    Args:
        x: Input of shape ``(..., 2 * d)``.

    Returns:
        Tensor of shape ``(..., d)``, matching
        :func:`unieinfra_kernel.contracts.activation.silu_and_mul_reference`.
    """
    return silu_and_mul_reference(x)
