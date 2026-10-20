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

"""Concrete ``build:`` capabilities for the in-tree kernels that need them.

These are the gates E30 had to leave open because ``min_arch_version`` cannot
express them. Each one answers "does the artifact installed *on this box*
contain code for *this* device", which is a property of the wheel/``.so``/JIT
toolchain, not of the silicon. See
:mod:`tokenspeed_kernel.artifact` for the mechanism.

Every probe here is **lazy** — nothing is imported or shelled out until a
selection actually needs the answer — and every registration declares
``default=True``, so if the probe itself cannot run we keep the
previously-validated behaviour instead of silently deleting a kernel on
hardware we cannot test (the E28 rule).
"""

from __future__ import annotations

from pathlib import Path

from tokenspeed_kernel.artifact import (
    binary_targets_device,
    register_build_capability,
    symbol_is_bound,
)

__all__ = [
    "DEEP_GEMM_DEVICE_CODE",
    "FLASHINFER_CUTLASS_MOE",
    "FLASHINFER_GEMM_FP8_BLOCKSCALE",
    "FLASHINFER_GDN_DECODE",
    "FLASHMLA_DEVICE_CODE",
    "MERGE_STATE_DEVICE_CODE",
]


def _sentinels() -> tuple:
    from tokenspeed_kernel.registry import error_fn

    return (error_fn,)


# ---------------------------------------------------------------------------
# DeepGEMM: JIT-compiled, ships per-arch-family implementation headers
# ---------------------------------------------------------------------------


def _probe_deep_gemm_device_code() -> bool:
    """Does the installed DeepGEMM have an implementation for this device?

    DeepGEMM JIT-compiles from headers named by architecture family
    (``sm90_*.cuh`` / ``sm100_*.cuh`` / ``sm120_*.cuh``), so the artifact
    question has three parts and none of them is an arch comparison: the package
    must bind, the header family for this device must be shipped, and the JIT's
    compiler must be reachable.

    E30 left ``deep_gemm_*`` unresolved precisely here. It recorded that
    DeepGEMM *does* ship ``sm120_fp8_fp4_gemm_1d1d.cuh``, so the sm_120 failure
    seen in that container was an NVCC/cccl **toolchain** failure and not a
    capability gap -- which is exactly the distinction this probe draws and an
    arch bound cannot.
    """
    import shutil

    import deep_gemm  # noqa: F401 - binding is part of the answer
    from tokenspeed_kernel.platform import current_platform

    if shutil.which("nvcc") is None:
        import os

        cuda_home = os.environ.get("CUDA_HOME")
        if not (cuda_home and (Path(cuda_home) / "bin" / "nvcc").exists()):
            return False

    impls = Path(deep_gemm.__file__).parent / "include" / "deep_gemm" / "impls"
    if not impls.is_dir():
        # Older layouts do not expose the headers; fall back to "it imported",
        # which is what the registration guard already relied on.
        return True

    arch = current_platform().arch_version
    families = {f"sm{arch.major}{arch.minor}", f"sm{arch.major}0"}
    return any(
        header.name.split("_", 1)[0] in families for header in impls.glob("sm*.cuh")
    )


DEEP_GEMM_DEVICE_CODE = register_build_capability(
    "build:deep_gemm_device_code",
    _probe_deep_gemm_device_code,
    default=True,
    doc="DeepGEMM binds, ships impl headers for this device's arch family, and "
    "has a reachable nvcc.",
)


# ---------------------------------------------------------------------------
# Prebuilt CUDA extension: ask the fatbinary directly
# ---------------------------------------------------------------------------


def _merge_state_object() -> str:
    from tokenspeed_kernel.thirdparty.cuda import merge_state as _merge_state_module

    return str(
        Path(_merge_state_module.__file__).parent
        / "objs"
        / "merge_state"
        / "merge_state.so"
    )


MERGE_STATE_DEVICE_CODE = register_build_capability(
    "build:merge_state_device_code",
    binary_targets_device(_merge_state_object),
    default=True,
    doc="The prebuilt merge_state.so fatbinary contains SASS (or "
    "forward-compatible PTX) for this device.",
)


# ---------------------------------------------------------------------------
# Optional third-party entry points: did the symbol actually bind?
# ---------------------------------------------------------------------------


def _load_flashinfer_gemm_fp8_blockscale():
    from tokenspeed_kernel.ops.gemm.flashinfer import gemm_fp8_nt_groupwise

    return gemm_fp8_nt_groupwise


FLASHINFER_GEMM_FP8_BLOCKSCALE = register_build_capability(
    "build:flashinfer_gemm_fp8_blockscale",
    symbol_is_bound(_load_flashinfer_gemm_fp8_blockscale, sentinels=_sentinels()),
    default=True,
    doc="FlashInfer's block-scaled fp8 GEMM entry point bound in this install.",
)


def _load_flashinfer_cutlass_fused_moe():
    from flashinfer import cutlass_fused_moe

    return cutlass_fused_moe


FLASHINFER_CUTLASS_MOE = register_build_capability(
    "build:flashinfer_cutlass_fused_moe",
    symbol_is_bound(_load_flashinfer_cutlass_fused_moe),
    default=True,
    doc="FlashInfer's CUTLASS fused-MoE entry point bound in this install.",
)


def _load_flashmla_with_kvcache():
    from tokenspeed_kernel.ops.attention.flash_mla import flash_mla_with_kvcache

    return flash_mla_with_kvcache


FLASHMLA_DEVICE_CODE = register_build_capability(
    "build:flashmla_with_kvcache",
    symbol_is_bound(_load_flashmla_with_kvcache, sentinels=_sentinels()),
    default=True,
    doc="FlashMLA's paged-KV decode entry point bound in this install.",
)


def _probe_flashinfer_gdn_decode() -> bool:
    from tokenspeed_kernel.ops.attention.flashinfer.gated_delta_rule import (
        is_decode_available,
    )

    return bool(is_decode_available())


FLASHINFER_GDN_DECODE = register_build_capability(
    "build:flashinfer_gdn_decode",
    _probe_flashinfer_gdn_decode,
    default=True,
    doc="FlashInfer's gated-delta-rule decode/MTP kernels are present in this "
    "install.",
)
