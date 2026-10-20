# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Per-operator input synthesis for autotuning and benchmarking.

Autotuning cannot time a kernel without *arguments*, and the arguments are an
operator property: a ``gemm.mm`` candidate wants ``A``/``B``/scales, an
attention candidate wants a paged KV cache. This module is the indirection that
keeps the selection core from knowing any of that — it is a registry of
callables keyed by ``(family, mode)``, populated by the op's contract module
(see :mod:`unieinfra_kernel.contracts.gemm`).

Two things are registered per operator:

``input generator``
    ``fn(*, format_signature, shape, device, seed) -> dict[str, Any]`` returning
    the keyword arguments for one call of any kernel registered for that
    operator. Every kernel in a ``(family, mode)`` shares one calling
    convention, which is exactly what makes them interchangeable, so one
    generator serves all candidates.

``benchmark shapes``
    Representative shape dicts to tune over when the caller does not supply a
    concrete shape. Normally these come from the operator's
    :class:`~unieinfra_kernel.contracts.KernelContract` ``workloads``.

Ported from ``tokenspeed_kernel.numerics.inputs`` (now under ``legacy/``), with
the class-based ``InputGenerator`` collapsed to a plain callable: nothing in the
autotune path used the class's state, and a function is one less thing for a
contract author to subclass.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "InputGeneratorFn",
    "ShapeTraitsFn",
    "get_benchmark_shapes",
    "get_input_generator",
    "get_shape_traits_fn",
    "list_generated_operators",
    "set_benchmark_shapes",
    "set_input_generator",
    "set_shape_traits_fn",
]

# fn(*, format_signature, shape, device, seed) -> kwargs for one kernel call.
InputGeneratorFn = Callable[..., dict[str, Any]]
# fn(shape) -> the trait dict runtime selection would pass for that shape.
ShapeTraitsFn = Callable[[dict[str, Any]], dict[str, Any] | None]

_INPUT_GENERATORS: dict[tuple[str, str], InputGeneratorFn] = {}
_BENCHMARK_SHAPES: dict[tuple[str, str], list[dict[str, Any]]] = {}
_SHAPE_TRAITS: dict[tuple[str, str], ShapeTraitsFn] = {}


def set_input_generator(
    family: str,
    mode: str,
    generator: InputGeneratorFn,
) -> None:
    """Register the input synthesizer for an operator.

    Args:
        family: Operator family, e.g. ``"gemm"``.
        mode: Operator mode, e.g. ``"mm"``.
        generator: Callable invoked as
            ``generator(format_signature=..., shape=..., device=..., seed=...)``
            returning the kwargs dict for one kernel call. It must raise
            ``ValueError``/``KeyError`` (not return garbage) for a signature or
            shape it cannot synthesize; autotuning treats that as "cannot
            measure this op" and falls back to heuristic ranking.
    """
    _INPUT_GENERATORS[(family, mode)] = generator


def get_input_generator(family: str, mode: str) -> InputGeneratorFn | None:
    """Return the registered input synthesizer, or ``None`` if there is none.

    Args:
        family: Operator family.
        mode: Operator mode.

    Returns:
        The generator callable, or ``None`` when the operator has not
        registered one (autotuning is then not possible for it).
    """
    return _INPUT_GENERATORS.get((family, mode))


def set_benchmark_shapes(
    family: str,
    mode: str,
    shapes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> None:
    """Register representative shapes to tune/benchmark an operator over.

    Args:
        family: Operator family.
        mode: Operator mode.
        shapes: Shape dicts. Keys are operator-specific (``M``/``N``/``K`` for
            GEMM); non-shape annotation keys such as ``label`` are preserved and
            ignored by the input generator.
    """
    _BENCHMARK_SHAPES[(family, mode)] = [dict(shape) for shape in shapes]


def get_benchmark_shapes(family: str, mode: str) -> list[dict[str, Any]]:
    """Return the registered representative shapes for an operator.

    Args:
        family: Operator family.
        mode: Operator mode.

    Returns:
        A fresh list of shape dicts; empty when none are registered.
    """
    return [dict(shape) for shape in _BENCHMARK_SHAPES.get((family, mode), ())]


def set_shape_traits_fn(family: str, mode: str, fn: ShapeTraitsFn) -> None:
    """Register how an operator turns a concrete shape into its trait dict.

    Startup warmup needs this so it resolves the *same* memo key runtime
    selection will: warming ``gemm.mm`` with no traits fills a cache entry no
    hot-path call ever looks up. The function is the operator's own
    ``shape_traits``, registered by its contract module.

    Args:
        family: Operator family.
        mode: Operator mode.
        fn: ``fn(shape) -> traits`` (or ``None`` when the shape is not one this
            operator recognises).
    """
    _SHAPE_TRAITS[(family, mode)] = fn


def get_shape_traits_fn(family: str, mode: str) -> ShapeTraitsFn | None:
    """Return the registered shape-to-traits function, or ``None``."""
    return _SHAPE_TRAITS.get((family, mode))


def list_generated_operators() -> list[tuple[str, str]]:
    """Return every ``(family, mode)`` that has an input generator registered."""
    return sorted(_INPUT_GENERATORS)
