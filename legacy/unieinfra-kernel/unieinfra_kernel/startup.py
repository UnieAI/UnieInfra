# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""The engine-facing startup hook: tune once, then freeze.

An engine calls :func:`autotune_startup` exactly once, after the model weights
are resident and before it starts serving. Everything about the design follows
from that single constraint:

* **Tuning is a startup activity, never a serving one.** It blocks the launch
  thread for the whole profiling run, and under tensor parallelism ranks that
  begin tuning at different times miss their next collective and deadlock. So
  this function ends by calling
  :func:`~unieinfra_kernel.selection.tuning.freeze_autotuning`, after which any
  shape first seen during serving is resolved heuristically — possibly a worse
  kernel, never a stall and never a hang.
* **The shapes come from the model, not from a table.** ``N`` and ``K`` are
  fixed the moment a ``Linear`` is built, so walking the loaded module tree
  yields exactly the projections that will run, including any a hand-maintained
  list would have forgotten. That is the same argument E34 measured: the
  declarative gate found ``down_proj`` and beat a hand-picked kernel list 4.2%
  to 1.6%.
* **It is opt-in and it is free when off.** With ``UNIEINFRA_AUTOTUNE`` unset,
  this only freezes tuning, which is a boolean assignment.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "autotune_enabled",
    "autotune_startup",
    "linear_shapes_from_model",
]

# Batch sizes to tune each projection at. The decode kernels all gate at
# ``m_le_8`` or tighter, so this covers every M where a first-party kernel can
# even be a candidate; every larger M resolves to the incumbent by trait gate
# alone and needs no measurement.
_DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8)

_TRUTHY = ("1", "true", "on", "yes")


def autotune_enabled() -> bool:
    """Whether the operator asked for measurement-based selection.

    Returns:
        ``True`` when ``UNIEINFRA_AUTOTUNE`` is set to a truthy value.
    """
    return os.environ.get("UNIEINFRA_AUTOTUNE", "0").strip().lower() in _TRUTHY


def linear_shapes_from_model(
    model: Any,
    *,
    batch_sizes: Iterable[int] = _DEFAULT_BATCH_SIZES,
    max_distinct: int = 32,
) -> list[dict[str, int]]:
    """Collect the distinct ``(N, K)`` of a loaded model's dense projections.

    Walks ``model.modules()`` for anything carrying a rank-2 ``weight``, which
    covers both ``torch.nn.Linear`` and vLLM's parallel-linear layers without
    importing either. Duplicate ``(N, K)`` pairs — 36 identical decoder layers
    give the same three projections 36 times — collapse to one entry, so the
    tuning cost is per *shape*, not per layer.

    Args:
        model: The loaded model module.
        batch_sizes: ``M`` values to pair with each ``(N, K)``.
        max_distinct: Safety cap on distinct ``(N, K)`` pairs, so a pathological
            model cannot make startup unbounded.

    Returns:
        Shape dicts ``{"M": ..., "N": ..., "K": ...}``, ordered by ``(N, K, M)``
        for a deterministic tuning order.
    """
    import torch

    seen: set[tuple[int, int]] = set()
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        N, K = int(weight.shape[0]), int(weight.shape[1])
        if N <= 0 or K <= 0:
            continue
        seen.add((N, K))
        if len(seen) >= max_distinct:
            logger.warning(
                "[unieinfra_kernel] autotune: stopping shape collection at %d "
                "distinct (N, K) pairs",
                max_distinct,
            )
            break

    return [
        {"M": m, "N": n, "K": k} for n, k in sorted(seen) for m in sorted(batch_sizes)
    ]


def autotune_startup(
    model: Any = None,
    *,
    shapes: list[dict[str, int]] | None = None,
    dtype: Any = None,
    freeze: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Tune kernel selection for this machine, then freeze tuning.

    Args:
        model: A loaded model to harvest shapes from. Ignored when ``shapes`` is
            given.
        shapes: Explicit shapes to tune, each ``{"M", "N", "K"}``. Falls back to
            the operator contract's own representative workloads when neither
            this nor ``model`` is supplied.
        dtype: Activation dtype the engine will run in. Defaults to the model's
            parameter dtype, or ``torch.bfloat16``.
        freeze: Freeze autotuning when finished. Leave ``True`` in an engine.
        force: Tune even when ``UNIEINFRA_AUTOTUNE`` is not set.

    Returns:
        A summary dict with ``enabled``, ``seconds``, ``shapes`` (count tuned),
        ``decisions`` (the autotune log) and ``disagreements`` (the subset where
        measurement overruled the heuristic — the interesting output).
    """
    from unieinfra_kernel.selection.selection import (
        SelectionStrategy,
        autotune_log,
        enable_autotuning,
        get_selection_policy,
        set_selection_policy,
        warmup_selection,
    )
    from unieinfra_kernel.selection.tuning import freeze_autotuning

    if not (force or autotune_enabled()):
        if freeze:
            freeze_autotuning()
        return {"enabled": False, "seconds": 0.0, "shapes": 0, "decisions": []}

    import torch
    from unieinfra_kernel.boundary import dense_tensor_format, format_signature

    if shapes is None:
        shapes = linear_shapes_from_model(model) if model is not None else None
    if dtype is None:
        dtype = _model_dtype(model) or torch.bfloat16

    previous_policy = get_selection_policy()
    enable_autotuning([("gemm", "mm")])
    try:
        if shapes is None:
            # No model handed over: tune the contract's own workloads.
            seconds = warmup_selection(freeze=False)
            tuned = len(autotune_log())
        else:
            from unieinfra_kernel.dispatch import shape_traits

            signature = format_signature(
                a=dense_tensor_format(dtype), b=dense_tensor_format(dtype)
            )
            ops = [
                (
                    "gemm",
                    "mm",
                    signature,
                    shape_traits(s["M"], s["N"], s["K"]),
                    s,
                )
                for s in shapes
            ]
            seconds = warmup_selection(ops, freeze=False)
            tuned = len(ops)
    except Exception:  # noqa: BLE001 - startup tuning must never block a launch
        logger.warning(
            "[unieinfra_kernel] autotune startup failed; keeping heuristic selection",
            exc_info=True,
        )
        set_selection_policy(previous_policy)
        if freeze:
            freeze_autotuning()
        return {"enabled": False, "seconds": 0.0, "shapes": 0, "decisions": []}

    decisions = autotune_log()
    disagreements = [d for d in decisions if d.get("disagreed")]

    # Keep the AUTOTUNE policy installed: after the freeze it resolves from the
    # disk cache when there is a hit and falls back to the heuristic when there
    # is not, which is exactly the intended serving behaviour.
    del previous_policy, SelectionStrategy

    if freeze:
        freeze_autotuning()

    logger.info(
        "[unieinfra_kernel] autotune: %d shapes in %.1fs, %d decision(s) "
        "disagreed with the heuristic%s",
        tuned,
        seconds,
        len(disagreements),
        (
            ": "
            + "; ".join(
                f"{d['shape']} {d['heuristic']} -> {d['measured']}"
                for d in disagreements[:8]
            )
            if disagreements
            else ""
        ),
    )
    return {
        "enabled": True,
        "seconds": seconds,
        "shapes": tuned,
        "decisions": decisions,
        "disagreements": disagreements,
    }


def _model_dtype(model: Any) -> Any:
    """First parameter dtype of ``model``, or ``None``."""
    if model is None:
        return None
    try:
        for param in model.parameters():
            return param.dtype
    except Exception:  # noqa: BLE001 - a model without parameters is fine
        return None
    return None
