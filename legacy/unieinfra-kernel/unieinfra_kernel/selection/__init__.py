# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Standalone capability-based kernel selection core.

This subpackage is the **base-independent** half of UnieInfra's kernel layer:
a registry of kernels declared with *capability* (vendor + arch + hardware
feature) and *trait* gates, plus a scoring pipeline that picks a winner for a
concrete request. It imports nothing but ``torch`` and the standard library, so
it runs unchanged inside SGLang-derived tokenspeed, inside stock vLLM, or in a
bare Python process.

Module map (ported verbatim from ``tokenspeed_kernel``, imports rewritten):

* :mod:`~unieinfra_kernel.selection.signature` — role-indexed tensor *format*
  signatures (dtype + quantization layout + scale granularity).
* :mod:`~unieinfra_kernel.selection.platform` — hardware detection and
  :class:`CapabilityRequirement`.
* :mod:`~unieinfra_kernel.selection.registry` — :class:`KernelSpec`,
  :class:`KernelRegistry`, :class:`Priority`, :func:`register_kernel`.
* :mod:`~unieinfra_kernel.selection.selection` — :func:`select_kernel` and the
  scoring/override machinery.

Runtime code should import through :mod:`unieinfra_kernel.boundary` rather than
from here directly, so that a process which already has ``tokenspeed_kernel``
loaded keeps using that single shared registry.
"""

from __future__ import annotations

from unieinfra_kernel.selection.artifact import (
    BuildCapability,
    binary_targets_device,
    build_capabilities,
    build_capability_available,
    clear_build_capability_cache,
    has_signature_parameter,
    kernel_is_registered,
    register_build_capability,
    symbol_is_bound,
)
from unieinfra_kernel.selection.autotune import (
    AutotuneCache,
    AutotuneParams,
    Measurement,
    autotune_cache,
    measure_candidates,
    shape_bucket,
    time_kernel,
)
from unieinfra_kernel.selection.inputs import (
    get_benchmark_shapes,
    get_input_generator,
    set_benchmark_shapes,
    set_input_generator,
    set_shape_traits_fn,
)
from unieinfra_kernel.selection.platform import (
    ArchVersion,
    CapabilityRequirement,
    Platform,
    PlatformInfo,
    current_platform,
)
from unieinfra_kernel.selection.registry import (
    KernelRegistry,
    KernelSpec,
    Priority,
    describe_kernel,
    load_builtin_kernels,
    register_kernel,
)
from unieinfra_kernel.selection.selection import (
    NoKernelFoundError,
    SelectedKernel,
    SelectionObjective,
    SelectionPin,
    SelectionPolicy,
    SelectionStrategy,
    autotune_log,
    clear_autotune_log,
    clear_kernel_pin,
    enable_autotuning,
    explain_selection,
    get_selection_policy,
    kernel_override,
    pin_kernel,
    select_kernel,
    set_kernel_pin,
    set_selection_policy,
    warmup_selection,
)
from unieinfra_kernel.selection.signature import (
    FormatSignature,
    ScaleFormat,
    TensorFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
    tensor_format,
)
from unieinfra_kernel.selection.tuning import (
    autotune_frozen,
    autotuning_unfrozen,
    freeze_autotuning,
    unfreeze_autotuning,
)

__all__ = [
    "ArchVersion",
    "AutotuneCache",
    "AutotuneParams",
    "Measurement",
    "SelectionPin",
    "SelectionPolicy",
    "SelectionStrategy",
    "autotune_cache",
    "autotune_frozen",
    "autotune_log",
    "autotuning_unfrozen",
    "clear_autotune_log",
    "clear_kernel_pin",
    "enable_autotuning",
    "freeze_autotuning",
    "get_benchmark_shapes",
    "get_input_generator",
    "get_selection_policy",
    "measure_candidates",
    "pin_kernel",
    "set_benchmark_shapes",
    "set_input_generator",
    "set_kernel_pin",
    "set_selection_policy",
    "set_shape_traits_fn",
    "shape_bucket",
    "time_kernel",
    "unfreeze_autotuning",
    "warmup_selection",
    "BuildCapability",
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
    "TensorFormat",
    "binary_targets_device",
    "build_capabilities",
    "build_capability_available",
    "clear_build_capability_cache",
    "current_platform",
    "dense_tensor_format",
    "describe_kernel",
    "explain_selection",
    "format_signature",
    "format_signatures",
    "has_signature_parameter",
    "kernel_is_registered",
    "kernel_override",
    "load_builtin_kernels",
    "register_build_capability",
    "register_kernel",
    "select_kernel",
    "symbol_is_bound",
    "tensor_format",
]
