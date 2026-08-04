# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""The single kernel-boundary import site for :mod:`unieinfra_kernel`.

Every module in this package imports registry / selection / platform /
signature symbols from *here* and never from a concrete implementation. That
indirection exists because the same kernels have to register into the same
registry on two different engine bases:

``tokenspeed`` base
    ``tokenspeed_kernel`` is installed and its ops façade (``gemm.mm`` and
    friends) is what the runtime calls. Registering into a second registry would
    make our kernels invisible, so we must use *its* registry — which is exactly
    the ``AGENTS.md`` rule that runtime code sees one kernel boundary.

``vLLM`` base (the re-base target)
    ``tokenspeed_kernel`` is not installed at all. The vendored standalone core
    in :mod:`unieinfra_kernel.selection` provides the identical API.

Backend choice, in order:

1. ``UNIEINFRA_KERNEL_REGISTRY=standalone|tokenspeed`` — explicit, wins.
2. ``tokenspeed_kernel`` already imported in this process → use it.
3. ``tokenspeed_kernel`` importable → use it.
4. Otherwise → the vendored standalone core.

:data:`BACKEND` records which one was chosen so tests and startup logs can
assert on it.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

__all__ = [
    "BACKEND",
    "ArchVersion",
    "CapabilityRequirement",
    "FormatSignature",
    "KernelRegistry",
    "KernelSpec",
    "NoKernelFoundError",
    "Platform",
    "PlatformInfo",
    "Priority",
    "ScaleFormat",
    "SelectedKernel",
    "SelectionObjective",
    "SelectionStrategy",
    "TensorFormat",
    "binary_targets_device",
    "build_capability_available",
    "current_platform",
    "dense_tensor_format",
    "describe_kernel",
    "explain_selection",
    "format_signature",
    "format_signatures",
    "has_signature_parameter",
    "kernel_is_registered",
    "register_build_capability",
    "register_kernel",
    "select_kernel",
    "symbol_is_bound",
    "tensor_format",
    "warmup_selection",
]


def _choose_backend() -> str:
    """Return ``"standalone"`` or ``"tokenspeed"`` for this process."""
    requested = os.environ.get("UNIEINFRA_KERNEL_REGISTRY", "auto").strip().lower()
    if requested in ("standalone", "tokenspeed"):
        return requested
    if requested not in ("", "auto"):
        logger.warning(
            "unieinfra_kernel: ignoring UNIEINFRA_KERNEL_REGISTRY=%r "
            "(expected 'auto', 'standalone' or 'tokenspeed')",
            requested,
        )
    if "tokenspeed_kernel" in sys.modules:
        return "tokenspeed"
    try:  # pragma: no cover - depends on what is installed
        import importlib.util

        if importlib.util.find_spec("tokenspeed_kernel") is not None:
            return "tokenspeed"
    except Exception:  # noqa: BLE001 - a broken install must not be fatal
        pass
    return "standalone"


BACKEND = _choose_backend()

if BACKEND == "tokenspeed":  # pragma: no cover - exercised on the tokenspeed base
    from tokenspeed_kernel.artifact import (  # noqa: F401
        binary_targets_device,
        build_capability_available,
        has_signature_parameter,
        kernel_is_registered,
        register_build_capability,
        symbol_is_bound,
    )
    from tokenspeed_kernel.platform import (  # noqa: F401
        ArchVersion,
        CapabilityRequirement,
        Platform,
        PlatformInfo,
        _get_cuda_sm_features,
        current_platform,
    )
    from tokenspeed_kernel.registry import (  # noqa: F401
        KernelRegistry,
        KernelSpec,
        Priority,
        describe_kernel,
        register_kernel,
    )
    from tokenspeed_kernel.selection import (  # noqa: F401
        NoKernelFoundError,
        SelectedKernel,
        SelectionObjective,
        SelectionStrategy,
        explain_selection,
    )
    from tokenspeed_kernel.selection import select_kernel as _tokenspeed_select_kernel
    from tokenspeed_kernel.selection import (  # noqa: F401
        spec_matches_traits,
        warmup_selection,
    )
    from tokenspeed_kernel.signature import (  # noqa: F401
        FormatSignature,
        ScaleFormat,
        TensorFormat,
        dense_tensor_format,
        format_signature,
        format_signatures,
        tensor_format,
    )
else:
    from unieinfra_kernel.selection.artifact import (  # noqa: F401
        binary_targets_device,
        build_capability_available,
        has_signature_parameter,
        kernel_is_registered,
        register_build_capability,
        symbol_is_bound,
    )
    from unieinfra_kernel.selection.platform import (  # noqa: F401
        ArchVersion,
        CapabilityRequirement,
        Platform,
        PlatformInfo,
        _get_cuda_sm_features,
        current_platform,
    )
    from unieinfra_kernel.selection.registry import (  # noqa: F401
        KernelRegistry,
        KernelSpec,
        Priority,
        describe_kernel,
        register_kernel,
    )
    from unieinfra_kernel.selection.selection import (  # noqa: F401
        NoKernelFoundError,
        SelectedKernel,
        SelectionObjective,
        SelectionStrategy,
        explain_selection,
        select_kernel,
        spec_matches_traits,
        warmup_selection,
    )
    from unieinfra_kernel.selection.signature import (  # noqa: F401
        FormatSignature,
        ScaleFormat,
        TensorFormat,
        dense_tensor_format,
        format_signature,
        format_signatures,
        tensor_format,
    )


if BACKEND == "tokenspeed":  # pragma: no cover - exercised on the tokenspeed base

    def select_kernel(*args, **kwargs):  # type: ignore[no-redef]
        """``tokenspeed_kernel.selection.select_kernel`` with our extra kwargs dropped.

        Measurement-based selection (``shape=``, ``strategy=``) lives in this
        package's standalone core; the tokenspeed base has its own, older
        selection module that does not accept those keywords. Rather than make
        every call site branch on :data:`BACKEND`, the boundary absorbs the
        difference: on the tokenspeed base the request is served heuristically,
        which is exactly what that base did before.

        Args:
            *args: Forwarded unchanged.
            **kwargs: Forwarded unchanged except ``shape`` and ``strategy``.

        Returns:
            The selected kernel.
        """
        kwargs.pop("shape", None)
        kwargs.pop("strategy", None)
        return _tokenspeed_select_kernel(*args, **kwargs)
