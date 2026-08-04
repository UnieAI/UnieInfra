# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

__all__ = [
    "ArchVersion",
    "InterconnectInfo",
    "PlatformInfo",
    "CapabilityRequirement",
    "Platform",
    "current_platform",
]

# Every vendor string the platform layer can report. A CapabilityRequirement
# that carries an arch bound must name a subset of these explicitly: AMD
# gfx950 is ArchVersion(9, 5) and CDNA5 is ArchVersion(12, 5) -- the very same
# numbers NVIDIA uses for Hopper-minor and workstation Blackwell -- so an arch
# bound without a vendor is latently cross-vendor and means nothing.
KNOWN_VENDORS = frozenset({"nvidia", "amd"})


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchVersion:
    """Hardware generation identifier. Supports comparison operators."""

    major: int
    minor: int

    def __ge__(self, other: ArchVersion) -> bool:
        return (self.major, self.minor) >= (other.major, other.minor)

    def __gt__(self, other: ArchVersion) -> bool:
        return (self.major, self.minor) > (other.major, other.minor)

    def __le__(self, other: ArchVersion) -> bool:
        return (self.major, self.minor) <= (other.major, other.minor)

    def __lt__(self, other: ArchVersion) -> bool:
        return (self.major, self.minor) < (other.major, other.minor)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class InterconnectInfo:
    """Multi-GPU interconnect topology."""

    topology: str  # "single_gpu", "pcie", "nvlink_pairs", "nvlink_full", "nvswitch"
    bandwidth_matrix: tuple[tuple[float, ...], ...] | None = None
    nvlink_version: int | None = None
    nvswitch_present: bool = False


@dataclass(frozen=True)
class Fp8E4M3FnDType:
    dtype: torch.dtype
    max: float
    min: float


_fp8e4m3_dtype = None


@dataclass(frozen=True)
class PlatformInfo:
    """Complete description of a compute platform."""

    vendor: str  # "nvidia", "amd"
    arch_version: ArchVersion
    device_name: str
    device_count: int

    # Memory
    total_memory: int  # Bytes per device
    memory_bandwidth: float  # GB/s

    # Compute
    sm_count: int  # Streaming multiprocessors (or CUs)
    max_threads_per_sm: int
    max_shared_memory_per_sm: int  # Bytes

    # Features (string-based for extensibility)
    sm_features: frozenset[str] = frozenset()  # Determined by compute capability
    runtime_features: frozenset[str] = frozenset()  # Detected at runtime

    # Interconnect (for multi-GPU)
    interconnect: InterconnectInfo | None = None

    # NUMA-local CPU IDs per logical device. Empty means unavailable.
    numa_cpu_affinity: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def detect(cls) -> PlatformInfo:
        """Detect platform from current environment."""
        return _detect_platform()

    # Convenience properties
    @property
    def is_nvidia(self) -> bool:
        return self.vendor == "nvidia"

    @property
    def is_hopper(self) -> bool:
        return self.is_nvidia and self.arch_version.major == 9

    @property
    def is_blackwell(self) -> bool:
        return self.is_nvidia and self.arch_version.major == 10

    @property
    def is_ampere(self) -> bool:
        return self.is_nvidia and self.arch_version.major == 8

    @property
    def is_amd(self) -> bool:
        return self.vendor == "amd"

    @property
    def is_cdna3(self) -> bool:
        return self.is_amd and self.arch_version == ArchVersion(9, 4)

    @property
    def is_cdna4(self) -> bool:
        return self.is_amd and self.arch_version == ArchVersion(9, 5)

    @property
    def is_cdna5(self) -> bool:
        return self.is_amd and self.arch_version == ArchVersion(12, 5)

    @property
    def is_ampere_plus(self) -> bool:
        return self.is_nvidia and self.arch_version >= ArchVersion(8, 0)

    @property
    def is_hopper_plus(self) -> bool:
        return self.is_nvidia and self.arch_version >= ArchVersion(9, 0)

    @property
    def is_blackwell_plus(self) -> bool:
        return self.is_nvidia and self.arch_version >= ArchVersion(10, 0)

    @property
    def has_tcgen05(self) -> bool:
        """Whether the device has tcgen05 tensor-core MMA and Tensor Memory.

        Returns:
            ``True`` when the detected SM feature set advertises
            ``compute:tcgen05`` (datacenter Blackwell: sm_100/sm_101/sm_103).

        Prefer this over ``is_blackwell_plus`` when gating a kernel that is
        built on tcgen05/TMEM. ``is_blackwell_plus`` is an open-ended version
        comparison that sm_120 also satisfies despite having no TMEM.
        """
        return "compute:tcgen05" in self.sm_features

    @property
    def has_cluster_launch(self) -> bool:
        """Whether the device supports thread-block clusters and distributed smem.

        Returns:
            ``True`` when the detected SM feature set advertises
            ``compute:cluster``.

        Sourced from the driver (``CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH``), not
        inferred from ``major >= 9``. The inference is wrong in *both*
        directions: sm_120 has clusters and distributed shared memory (verified
        with ``ptxas`` and the driver in E30) even though it is not a datacenter
        part, while the H20 SKU reports sm_90 and does **not** support cluster
        launch -- which used to need a device-name regex to work around.
        """
        return "compute:cluster" in self.sm_features

    @property
    def has_fp4_tensor_cores(self) -> bool:
        """Whether the device has native FP4 (nvfp4/mxfp4) tensor cores.

        Returns:
            ``True`` when the detected SM feature set advertises
            ``tensor_core:f4``.

        Prefer this over ``is_blackwell``/``is_blackwell_plus`` when gating an
        FP4 kernel. ``is_blackwell`` tests ``major == 10`` exactly, which
        excludes sm_120 (RTX PRO / workstation Blackwell) even though sm_120
        *does* have FP4 tensor cores; ``is_blackwell_plus`` is an open-ended
        version comparison that says nothing about the capability it stands in
        for. This property asks the detected capability set directly and is
        vendor-neutral (AMD gfx950/gfx1250 also advertise ``tensor_core:f4``).
        """
        return "tensor_core:f4" in self.sm_features

    @property
    def is_cdna3_plus(self) -> bool:
        return self.is_amd and self.arch_version >= ArchVersion(9, 4)

    @property
    def is_cdna4_plus(self) -> bool:
        return self.is_amd and self.arch_version >= ArchVersion(9, 5)

    @property
    def is_cdna5_plus(self) -> bool:
        return self.is_amd and self.arch_version >= ArchVersion(12, 5)

    @property
    def arch(self) -> str:
        """Short architecture string for cache keys."""
        return str(self.arch_version)

    @property
    def is_fp8e4m3fnuz(self) -> bool:
        return self.is_cdna3

    @property
    def fp8e4m3fn(self) -> Fp8E4M3FnDType:
        global _fp8e4m3_dtype
        if _fp8e4m3_dtype is None:
            if self.is_cdna3:
                dtype = torch.float8_e4m3fnuz
                fp8_max = 224.0
            else:
                dtype = torch.float8_e4m3fn
                fp8_max = torch.finfo(dtype).max
            fp8_min = -fp8_max
            _fp8e4m3_dtype = Fp8E4M3FnDType(dtype=dtype, max=fp8_max, min=fp8_min)
        return _fp8e4m3_dtype

    def register_host_tensor_for_gpu_access(self, tensor: torch.Tensor) -> None:
        """Register host memory that GPU kernels will directly dereference."""
        if tensor.device.type != "cpu" or tensor.numel() == 0:
            return
        status = torch.cuda.cudart().cudaHostRegister(
            tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0
        )
        if int(status) != 0:
            raise RuntimeError(f"cudaHostRegister failed with {status!s}")

    def device_visible_data_ptr(self, tensor: torch.Tensor) -> int:
        """Return a pointer value that is valid to dereference from GPU kernels."""
        ptr = tensor.data_ptr()
        if self.is_amd and tensor.device.type == "cpu" and tensor.numel() > 0:
            return _hip_host_get_device_pointer(ptr)
        return ptr

    @property
    def generation_name(self) -> str:
        """Human-readable generation name."""
        arch_version = (self.arch_version.major, self.arch_version.minor)
        if self.is_nvidia:
            names = {
                (8, 0): "Ampere",
                (8, 6): "Ampere",
                (8, 9): "Ada Lovelace",
                (9, 0): "Hopper",
                (10, 0): "Blackwell",
            }
            return names.get(arch_version, f"SM{arch_version[0]}.{arch_version[1]}")
        if self.is_amd:
            names = {
                (9, 4): "CDNA3",  # MI300
                (9, 5): "CDNA4",  # MI350
                (12, 5): "CDNA5",
            }
            return names.get(arch_version, f"GFX{arch_version[0]}.{arch_version[1]}")
        return f"{self.vendor}:{arch_version[0]}.{arch_version[1]}"


_OPEN_ENDED_HELP = """\
A `min_arch_version` on its own is not a capability -- it is an open-ended
version comparison, and every part newer than the one you tested satisfies it.
That is exactly how four kernels won selection on sm_120 and then hard-failed at
runtime (E30), and how upstream's `ll_bf16` CuteDSL GEMM killed EngineCore at
warmup on sm_120 while gating on `has_device_capability(90)` (E37).

Add ONE of:

  * `required_features={...}` -- the capability the version number was standing
    in for. Prefer this: it classifies a part we have never seen correctly by
    construction. Hardware features come from the device
    (`compute:tcgen05`, `compute:cluster`, `tensor_core:f4`,
    `tensor_core:block_scale`, `memory:tma`, `memory:async_copy`, ...);
    `build:*` features come from the artifact-probe registry in
    `unieinfra_kernel.selection.artifact` and answer "does the installed binary
    contain code for this device", which no version number can express.

  * `max_arch_version=...` -- an explicit upper bound, when the kernel really is
    restricted to a known range of parts and you accept that a newer part must
    be re-qualified by hand.
"""


@dataclass(frozen=True)
class CapabilityRequirement:
    """Requirements a kernel has on platform capabilities.

    Args:
        min_arch_version: Lowest architecture the kernel supports. May not be
            given alone -- see below.
        max_arch_version: Highest architecture the kernel supports (inclusive).
        required_features: Capability names the device must advertise. Hardware
            features are matched against ``PlatformInfo.sm_features`` and
            ``runtime_features``; names in the ``build:`` namespace are resolved
            through :mod:`unieinfra_kernel.selection.artifact`.
        vendors: Vendors the kernel supports. ``None`` means *any vendor*, which
            is only legal for a requirement that carries no arch bound.

    Raises:
        ValueError: If ``min_arch_version`` is supplied with neither a
            ``max_arch_version`` nor ``required_features`` (the mis-gating bug
            class this type exists to prevent); if an arch bound is supplied
            without an explicit ``vendors``; if ``max_arch_version`` is below
            ``min_arch_version``; or if ``vendors`` names an unknown vendor.
    """

    min_arch_version: ArchVersion | None = None
    max_arch_version: ArchVersion | None = None
    required_features: frozenset[str] = frozenset()
    vendors: frozenset[str] | None = None  # None = any vendor

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "required_features", frozenset(self.required_features or ())
        )
        if self.vendors is not None:
            object.__setattr__(self, "vendors", frozenset(self.vendors))

        has_arch_bound = (
            self.min_arch_version is not None or self.max_arch_version is not None
        )

        if self.vendors is not None:
            unknown = self.vendors - KNOWN_VENDORS
            if unknown:
                raise ValueError(
                    f"CapabilityRequirement: unknown vendor(s) {sorted(unknown)}. "
                    f"Known vendors: {sorted(KNOWN_VENDORS)}."
                )
            if not self.vendors:
                raise ValueError(
                    "CapabilityRequirement: `vendors=frozenset()` can never be "
                    "satisfied. Omit `vendors` for a vendor-neutral requirement, "
                    "or name the vendors the kernel supports."
                )

        # Rule 1 (design principles P1): a min bound with no upper bound and no
        # feature requirement is the mis-gating form. Make it unrepresentable.
        if (
            self.min_arch_version is not None
            and self.max_arch_version is None
            and not self.required_features
        ):
            raise ValueError(
                f"CapabilityRequirement(min_arch_version={self.min_arch_version}) "
                "needs either `required_features` or `max_arch_version`.\n\n"
                + _OPEN_ENDED_HELP
            )

        # Rule 4: an arch bound is meaningless without a vendor, because the
        # vendors share the numbers.
        if has_arch_bound and self.vendors is None:
            raise ValueError(
                "CapabilityRequirement: an arch bound requires an explicit "
                "`vendors=`.\n\n"
                "ArchVersion numbers are NOT vendor-unique: AMD gfx950 (CDNA4) is "
                "ArchVersion(9, 5) and CDNA5 is ArchVersion(12, 5) -- the same "
                "numbers NVIDIA uses for Hopper-minor and workstation Blackwell. "
                "A bound such as `min_arch_version=ArchVersion(9, 0)` therefore "
                "admits every CDNA part as well. Pass "
                "`vendors=frozenset({'nvidia'})` (or the vendors you actually "
                "support). A requirement with no arch bound at all may stay "
                "vendor-neutral."
            )

        if (
            self.min_arch_version is not None
            and self.max_arch_version is not None
            and self.max_arch_version < self.min_arch_version
        ):
            raise ValueError(
                "CapabilityRequirement: max_arch_version "
                f"{self.max_arch_version} is below min_arch_version "
                f"{self.min_arch_version}; nothing can satisfy it."
            )

    def satisfied_by(self, platform: PlatformInfo) -> bool:
        """Check if platform satisfies these requirements."""
        if self.vendors and platform.vendor not in self.vendors:
            return False

        if self.min_arch_version:
            if not platform.arch_version >= self.min_arch_version:
                return False

        if self.max_arch_version:
            if platform.arch_version > self.max_arch_version:
                return False

        return not self.missing_features(platform)

    def missing_features(self, platform: PlatformInfo) -> set[str]:
        """Return features required but not available.

        Hardware features are looked up in the platform's detected feature sets.
        Features in the ``build:`` namespace describe the *installed artifact*
        rather than the silicon, so they are resolved through
        :func:`unieinfra_kernel.selection.artifact.build_capability_available`,
        which probes lazily and caches.
        """
        from unieinfra_kernel.selection.artifact import (
            BUILD_NAMESPACE,
            build_capability_available,
        )

        all_features = platform.sm_features | platform.runtime_features
        missing = self.required_features - all_features
        return {
            feature
            for feature in missing
            if not (
                feature.startswith(BUILD_NAMESPACE)
                and build_capability_available(feature)
            )
        }


# ---------------------------------------------------------------------------
# Platform singleton
# ---------------------------------------------------------------------------


class Platform:
    """Global platform singleton with lazy initialization."""

    _instance: PlatformInfo | None = None

    @classmethod
    def get(cls) -> PlatformInfo:
        """Get current platform info (detected once, cached)."""
        if cls._instance is None:
            cls._instance = PlatformInfo.detect()
        return cls._instance

    @classmethod
    def override(cls, platform: PlatformInfo) -> None:
        """Override platform detection (for testing/debugging)."""
        cls._instance = platform

    @classmethod
    def reset(cls) -> None:
        """Reset cached platform (for testing)."""
        cls._instance = None
        _detect_cuda_nvlink_topology.cache_clear()


def current_platform() -> PlatformInfo:
    """Get current platform."""
    return Platform.get()


# ---------------------------------------------------------------------------
# Detection implementation
# ---------------------------------------------------------------------------


def _torch_version() -> tuple[int, ...]:
    """Return PyTorch version as a comparable tuple, e.g. (2, 7, 0)."""
    try:
        import torch

        return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
    except Exception:
        return (0, 0, 0)


def _detect_platform() -> PlatformInfo:
    """Detect current platform capabilities."""
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "unieinfra-kernel requires PyTorch with NVIDIA CUDA or AMD ROCm support."
        ) from None

    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip:
            return _detect_rocm_platform()
        return _detect_cuda_platform()

    raise RuntimeError("unieinfra-kernel requires an NVIDIA CUDA or AMD ROCm GPU.")


def _detect_cuda_platform() -> PlatformInfo:
    """Detect NVIDIA CUDA platform."""
    import torch

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    arch_version = ArchVersion(props.major, props.minor)
    sm_features = _get_cuda_sm_features(
        arch_version,
        cluster_launch=_detect_cuda_cluster_launch(device_index),
        device_name=props.name,
    )
    runtime_features = _get_cuda_runtime_features()
    interconnect = _detect_cuda_interconnect()
    numa_cpu_affinity = _detect_cuda_numa_cpu_affinity()

    return PlatformInfo(
        vendor="nvidia",
        arch_version=arch_version,
        device_name=props.name,
        device_count=torch.cuda.device_count(),
        total_memory=props.total_memory,
        memory_bandwidth=_estimate_bandwidth(props),
        sm_count=props.multi_processor_count,
        max_threads_per_sm=getattr(props, "max_threads_per_multi_processor", 0),
        max_shared_memory_per_sm=_cuda_shared_memory_per_block(props),
        sm_features=sm_features,
        runtime_features=runtime_features,
        interconnect=interconnect,
        numa_cpu_affinity=numa_cpu_affinity,
    )


def _cuda_shared_memory_per_block(props: object) -> int:
    """Largest shared-memory allocation a single CUDA block can opt into.

    Args:
        props: The ``torch.cuda.get_device_properties()`` record for the device.

    Returns:
        Opt-in shared memory per block, in bytes, or ``0`` when the running
        torch build exposes none of the known attributes.

    ``torch.cuda`` has never exposed ``max_shared_memory_per_block``, so the
    previous lookup silently reported ``0`` on every NVIDIA device. Kernels that
    size their tiles from this budget need the real number: datacenter parts
    (sm_90/sm_100) offer ~228 KB per block while sm_86/sm_89 and sm_120
    (RTX PRO / workstation Blackwell) offer only ~100 KB.
    """
    for attr in (
        "max_shared_memory_per_block",
        "shared_memory_per_block_optin",
        "shared_memory_per_multiprocessor",
        "shared_memory_per_block",
    ):
        value = getattr(props, attr, 0) or 0
        if value:
            return int(value)
    return 0


#: ``CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH`` from ``cuda.h``. Reported as 1 on
#: sm_90 (H100/H200), sm_100/sm_103 and sm_120; 0 on the H20 SKU, which is an
#: sm_90 part that cannot launch clusters.
_CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120

#: Device-name patterns for SKUs that report an architecture whose generation has
#: cluster launch but that do not themselves support it. Consulted ONLY when the
#: driver could not be queried at all, so the pre-driver-query behaviour is
#: preserved exactly rather than regressing to the plain version inference.
#: Matched on word boundaries -- "H20" must not match "H200".
_NO_CLUSTER_LAUNCH_SKUS = (r"\bH20\b",)


def _detect_cuda_cluster_launch(device_index: int) -> bool | None:
    """Ask the driver whether this device supports thread-block cluster launch.

    Args:
        device_index: Ordinal of the CUDA device to query.

    Returns:
        ``True``/``False`` from ``cuDeviceGetAttribute`` with
        ``CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH``, or ``None`` when the driver
        cannot be loaded or the query fails -- in which case the caller falls
        back to the previously-validated inference rather than narrowing.
    """
    for lib_name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    else:
        return None

    try:
        # cuInit is idempotent and cheap; torch has normally already run it.
        if lib.cuInit(0) != 0:
            return None
        device = ctypes.c_int()
        if lib.cuDeviceGet(ctypes.byref(device), ctypes.c_int(device_index)) != 0:
            return None
        value = ctypes.c_int()
        status = lib.cuDeviceGetAttribute(
            ctypes.byref(value),
            ctypes.c_int(_CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH),
            device,
        )
        if status != 0:
            return None
        return bool(value.value)
    except Exception as exc:  # noqa: BLE001 - detection must never be fatal
        logger.warning("Driver query for cluster-launch support failed: %s", exc)
        return None


def _get_cuda_sm_features(
    arch_version: ArchVersion,
    *,
    cluster_launch: bool | None = None,
    device_name: str = "",
) -> frozenset[str]:
    """Determine CUDA SM features for a device.

    Args:
        arch_version: The device's compute capability.
        cluster_launch: Driver-reported cluster-launch support, from
            :func:`_detect_cuda_cluster_launch`. ``None`` means the driver could
            not be asked, and the previously-validated inference is used
            instead so a detection gap never narrows a working path (E28).
        device_name: Device name, consulted only on the ``None`` path to keep
            the known-bad-SKU behaviour that predates the driver query.

    Returns:
        The device's hardware feature set. Everything except ``compute:cluster``
        comes from an explicit arch -> feature table; ``compute:cluster`` comes
        from the driver, because the version inference is wrong in both
        directions (sm_120 *has* clusters; H20 reports sm_90 and does not).
    """
    features: set[str] = set()

    if arch_version >= ArchVersion(7, 0):
        features |= {"tensor_core:f16"}

    if arch_version >= ArchVersion(8, 0):
        features |= {"tensor_core:int8", "memory:async_copy"}

    if arch_version >= ArchVersion(8, 9):
        features |= {"tensor_core:f8"}

    if arch_version >= ArchVersion(9, 0):
        features |= {"memory:tma"}

    if cluster_launch is None:
        import re

        cluster_launch = arch_version >= ArchVersion(9, 0) and not any(
            re.search(pattern, device_name, re.IGNORECASE)
            for pattern in _NO_CLUSTER_LAUNCH_SKUS
        )
    if cluster_launch:
        features |= {"compute:cluster"}

    if arch_version >= ArchVersion(10, 0):
        # Blackwell adds native FP4 tensor cores and, with them, block-scaled
        # MMA (a scale-factor operand per K block: nvfp4, mxfp8, mxfp4). Both
        # are present on datacenter Blackwell AND on sm_120 -- unlike tcgen05.
        features |= {"tensor_core:f4", "tensor_core:block_scale"}

    # tcgen05 -- the 5th-generation tensor-core MMA family plus its Tensor
    # Memory (TMEM) -- exists only on *datacenter* Blackwell (sm_100/sm_101/
    # sm_103). It is deliberately NOT an open-ended ">= (10, 0)" comparison:
    # sm_120 (RTX PRO / workstation Blackwell) is arch 12.0 and would satisfy
    # such a check, yet it has no TMEM at all.
    #
    # VERIFIED with ptxas (CUDA 12.9): `tcgen05.alloc` assembles for
    # `sm_100a` and is rejected for both `sm_90a` and `sm_120a`
    # ("Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'").
    #
    # Kernels built on tcgen05/TMEM (trtllm-gen FMHA and MoE, CUTLASS sm100,
    # the CuTe DSL sm100 kernels) should require this feature instead of
    # writing `min_arch_version=(10, 0)` with no upper bound.
    if ArchVersion(10, 0) <= arch_version < ArchVersion(11, 0):
        features |= {"compute:tcgen05"}

    return frozenset(features)


def _get_cuda_runtime_features() -> frozenset[str]:
    """Detect CUDA runtime features from environment."""
    features: set[str] = {"runtime:cuda_graph"}

    if _check_symmetric_memory_available():
        features.add("comms:symmetric_memory")
    if _check_nvlink_available():
        features.add("comms:nvlink")
    if _detect_cuda_nvlink_topology() == "nvlink_full":
        features.add("comms:nvlink_full")

    return frozenset(features)


def _detect_rocm_platform() -> PlatformInfo:
    """Detect AMD ROCm platform."""
    import torch

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = _extract_amd_arch(props.gcnArchName)

    # Map AMD architectures
    arch_map = {
        "gfx942": ArchVersion(9, 4),  # MI300
        "gfx950": ArchVersion(9, 5),  # MI350
        "gfx1250": ArchVersion(12, 5),
    }
    arch_version = arch_map.get(arch, ArchVersion(9, 0))
    sm_features = _get_rocm_sm_features(arch)
    runtime_features = _get_rocm_runtime_features()

    return PlatformInfo(
        vendor="amd",
        arch_version=arch_version,
        device_name=props.name,
        device_count=torch.cuda.device_count(),
        total_memory=props.total_memory,
        memory_bandwidth=_estimate_amd_bandwidth(props),
        sm_count=props.multi_processor_count,
        max_threads_per_sm=getattr(props, "max_threads_per_multi_processor", 0),
        max_shared_memory_per_sm={
            "gfx942": 64 * 1024,
            "gfx950": 160 * 1024,
        }.get(arch, getattr(props, "max_shared_memory_per_block", 0)),
        sm_features=sm_features,
        runtime_features=runtime_features,
        interconnect=_detect_rocm_interconnect(),
    )


def _get_rocm_sm_features(arch: str) -> frozenset[str]:
    """Determine ROCm SM features from architecture."""
    features: set[str] = set()

    if arch in ("gfx942", "gfx950", "gfx1250"):
        features |= {"tensor_core:f16", "tensor_core:f8"}

    if arch in ("gfx950", "gfx1250"):
        # CDNA4/CDNA5 MFMA takes per-block scale factors (mxfp4/mxfp8), the same
        # capability NVIDIA gained at sm_100 -- so the feature name is shared and
        # a `vendors=` gate, not the feature, is what keeps kernels apart.
        features |= {"tensor_core:f4", "tensor_core:block_scale", "memory:async_copy"}

    return frozenset(features)


def _get_rocm_runtime_features() -> frozenset[str]:
    """Detect ROCm runtime features from environment."""
    features: set[str] = set()

    if _check_symmetric_memory_available():
        features.add("comms:symmetric_memory")

    return frozenset(features)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_amd_arch(gcn_arch_name: str) -> str:
    """Extract base architecture from GCN arch name.

    Example: 'gfx942:sramecc+:xnack-' -> 'gfx942'
    """
    return gcn_arch_name.split(":")[0]


def _estimate_bandwidth(props: object) -> float:
    """Estimate memory bandwidth in GB/s from CUDA device properties."""
    clock_rate = getattr(props, "memory_clock_rate", 0)
    bus_width = getattr(props, "memory_bus_width", 0)
    if clock_rate and bus_width:
        return (clock_rate * 1e3 * (bus_width / 8) * 2) / 1e9
    return 0.0


def _estimate_amd_bandwidth(props: object) -> float:
    """Estimate memory bandwidth for AMD devices."""
    clock_rate = getattr(props, "memory_clock_rate", 0)
    bus_width = getattr(props, "memory_bus_width", 0)
    if clock_rate and bus_width:
        return (clock_rate * 1e3 * (bus_width / 8) * 2) / 1e9
    return 0.0


def _detect_cuda_interconnect() -> InterconnectInfo | None:
    """Detect CUDA multi-GPU interconnect topology."""
    try:
        import torch

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return InterconnectInfo(topology="single_gpu")

        nvlink_topology = _detect_cuda_nvlink_topology()
        if nvlink_topology:
            return InterconnectInfo(topology=nvlink_topology)
        return InterconnectInfo(topology="pcie")
    except Exception:
        return None


def _detect_rocm_interconnect() -> InterconnectInfo | None:
    """Detect ROCm multi-GPU interconnect topology."""
    try:
        import torch

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return InterconnectInfo(topology="single_gpu")
        # Probe /sys/class/kfd for xGMI links (HSA_IOLINK_TYPE_XGMI = 11).
        try:
            import os as _os

            kfd_root = "/sys/class/kfd/kfd/topology/nodes"
            xgmi_count = 0
            for node in _os.listdir(kfd_root):
                links_dir = _os.path.join(kfd_root, node, "io_links")
                if not _os.path.isdir(links_dir):
                    continue
                for link in _os.listdir(links_dir):
                    pf = _os.path.join(links_dir, link, "properties")
                    try:
                        with open(pf) as f:
                            for line in f:
                                if line.startswith("type ") and line.split()[1] == "11":
                                    xgmi_count += 1
                    except OSError:
                        continue
            if xgmi_count > 0:
                full = device_count * (device_count - 1)
                topo = "xgmi_full" if xgmi_count >= full else "xgmi_pairs"
                return InterconnectInfo(topology=topo)
        except Exception:
            pass
        return InterconnectInfo(topology="pcie")
    except Exception:
        return None


def _detect_cuda_numa_cpu_affinity() -> tuple[tuple[int, ...], ...]:
    """Return NUMA-local CPU IDs per visible CUDA device using NVML."""
    nvml_initialized = False
    try:
        import pynvml

        device_count = torch.cuda.device_count()
        if device_count == 0:
            return ()

        pynvml.nvmlInit()
        nvml_initialized = True

        c_ulong_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        cpu_count = os.cpu_count()
        if not cpu_count:
            return ()

        affinities: list[tuple[int, ...]] = []
        for device_id in range(device_count):
            props = torch.cuda.get_device_properties(device_id)
            pci_bus_id = (
                f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:"
                f"{props.pci_device_id:02X}.0"
            )
            handle = pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id)
            masks = pynvml.nvmlDeviceGetCpuAffinity(
                handle, math.ceil(cpu_count / c_ulong_bits)
            )
            affinities.append(
                tuple(
                    cpu
                    for cpu in range(cpu_count)
                    if masks[cpu // c_ulong_bits] & (1 << (cpu % c_ulong_bits))
                )
            )
    except Exception as e:
        logger.warning("NVML failed to query NUMA affinity: %s", e)
        return ()
    finally:
        if nvml_initialized:
            pynvml.nvmlShutdown()

    return tuple(affinities)


@lru_cache(maxsize=1)
def _detect_cuda_nvlink_topology() -> str | None:
    """Return NVLink topology for visible CUDA devices using NVML."""
    nvml_initialized = False
    try:
        import pynvml

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return None

        pynvml.nvmlInit()
        nvml_initialized = True

        handles = []
        for device_id in range(device_count):
            props = torch.cuda.get_device_properties(device_id)
            pci_bus_id = (
                f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:"
                f"{props.pci_device_id:02X}.0"
            )
            handles.append(pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id))

        has_nvlink = False
        full_nvlink = True
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                try:
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                    if p2p_status == pynvml.NVML_P2P_STATUS_OK:
                        has_nvlink = True
                    else:
                        full_nvlink = False
                except pynvml.NVMLError:
                    full_nvlink = False
    except Exception as e:
        logger.warning("NVML failed to query NVLink topology: %s", e)
        return None
    finally:
        if nvml_initialized:
            pynvml.nvmlShutdown()

    if full_nvlink:
        return "nvlink_full"
    if has_nvlink:
        return "nvlink_pairs"
    return None


def _check_symmetric_memory_available() -> bool:
    """Check if PyTorch symmetric memory is available."""
    try:
        import torch.distributed._symmetric_memory  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


def _check_nvlink_available() -> bool:
    """Check if NVLink connectivity is available."""
    return _detect_cuda_nvlink_topology() is not None


@lru_cache(maxsize=1)
def _get_hip_runtime():
    lib_name = "libamdhip64.so"
    candidates = []
    torch_hip_path = Path(torch.__file__).resolve().parent / "lib" / lib_name
    if torch_hip_path.exists():
        candidates.append(str(torch_hip_path))
    candidates.append(lib_name)

    last_error = None
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
            lib.hipHostGetDevicePointer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            lib.hipHostGetDevicePointer.restype = ctypes.c_int
            if hasattr(lib, "hipGetErrorString"):
                lib.hipGetErrorString.argtypes = [ctypes.c_int]
                lib.hipGetErrorString.restype = ctypes.c_char_p
            return lib
        except OSError as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load {lib_name}") from last_error


def _hip_host_get_device_pointer(host_ptr: int) -> int:
    lib = _get_hip_runtime()
    device_ptr = ctypes.c_void_p()
    error = lib.hipHostGetDevicePointer(
        ctypes.byref(device_ptr), ctypes.c_void_p(host_ptr), 0
    )
    if error != 0:
        error_str = f"HIP error {error}"
        if hasattr(lib, "hipGetErrorString"):
            raw_error_str = lib.hipGetErrorString(error)
            if raw_error_str:
                error_str = raw_error_str.decode()
        raise RuntimeError(
            "hipHostGetDevicePointer failed for registered host pointer "
            f"0x{host_ptr:x}: {error_str}"
        )
    if device_ptr.value is None:
        raise RuntimeError(
            f"hipHostGetDevicePointer returned null for registered host pointer 0x{host_ptr:x}"
        )
    return device_ptr.value
