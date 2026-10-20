# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""The dense ``gemm.mm`` incumbent for the *standalone* registry.

On the tokenspeed base this role is played by ``torch_mm``
(``tokenspeed_kernel/numerics/reference/gemm.py``, ``PORTABLE + 3``). When
:mod:`unieinfra_kernel` runs standalone — i.e. inside vLLM, where
``tokenspeed_kernel`` is not installed — that kernel is not present, and without
an incumbent :func:`select_kernel` would either raise for every shape outside
the decode slice or have nothing for our kernel to *beat*.

So this module registers the same thing at the same priority band: a thin
``F.linear``. Two consequences that matter:

* Selection stays a real decision. For ``M <= 2`` the decode GEMV
  (``SPECIALIZED + 1`` = 13) outranks this (``PORTABLE + 3`` = 7); for every
  other shape the ``m_le_8`` / ``n_le_16384`` trait gate removes the GEMV from
  the candidate set entirely and this wins. Both outcomes are observable with
  ``UNIEINFRA_KERNEL_VERBOSE=1``.
* Outside the gated slice the selected implementation *is* ``F.linear``, i.e.
  bit-identical to what stock vLLM's ``UnquantizedLinearMethod.apply`` calls
  (``vllm/model_executor/layers/utils.py``: the CUDA arm of
  ``dispatch_unquantized_gemm`` is a hardwired ``F.linear``).

Registration is skipped when the boundary resolved to the ``tokenspeed``
registry, because that registry already has ``torch_mm`` at the identical
priority and adding a second one there could reorder an existing tie.
"""

from __future__ import annotations

import torch
from unieinfra_kernel.boundary import (
    BACKEND,
    Priority,
    dense_tensor_format,
    format_signature,
    register_kernel,
)

__all__ = ["torch_linear"]

_DENSE_SIGNATURES = frozenset(
    format_signature(a=dense_tensor_format(dtype), b=dense_tensor_format(dtype))
    for dtype in (torch.bfloat16, torch.float16, torch.float32)
)


def torch_linear(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense ``A @ B.T (+ bias)`` via :func:`torch.nn.functional.linear`.

    Args:
        A: Activations ``[M, K]``.
        B: Weights ``[N, K]`` (``torch.nn.Linear`` layout).
        A_scales: Unused on the dense path.
        B_scales: Unused on the dense path.
        out_dtype: Requested output dtype.
        alpha: Optional global output scale.
        block_size: Unused on the dense path.
        bias: Optional ``[N]`` bias, applied by ``F.linear`` itself.
        out: Optional ``[M, N]`` destination, copied into when supplied.

    Returns:
        ``[M, N]`` tensor (``out`` when supplied).
    """
    del A_scales, B_scales, block_size
    result = torch.nn.functional.linear(A, B, bias)
    if alpha is not None:
        result = result * alpha.to(dtype=result.dtype)
    if result.dtype != out_dtype:
        result = result.to(out_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


if BACKEND == "standalone":
    torch_linear = register_kernel(  # noqa: F811 - decorator applied conditionally
        "gemm",
        "mm",
        name="torch_linear",
        solution="reference",
        signatures=_DENSE_SIGNATURES,
        traits={},
        priority=Priority.PORTABLE + 3,
        tags={"determinism", "portability"},
        origin="vendor",
    )(torch_linear)
