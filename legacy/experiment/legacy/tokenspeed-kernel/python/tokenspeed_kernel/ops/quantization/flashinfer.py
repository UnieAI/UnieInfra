# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

fp4_quantize = error_fn
flashinfer_quantize_mxfp8 = error_fn
flashinfer_quantize_nvfp4 = error_fn
mxfp8_quantize = error_fn
nvfp4_block_scale_interleave = error_fn
fp8_blockscale_quantize_runner_sm90 = error_fn

if platform.is_nvidia:
    from flashinfer import mxfp8_quantize

    if platform.is_hopper:
        from flashinfer.gemm.gemm_base import (
            get_fp8_blockscale_gemm_runner_sm90 as fp8_blockscale_quantize_runner_sm90,
        )

    # FlashInfer's MXFP8 quantizer supports SM100 and newer architectures.
    # Hopper uses the portable Triton implementation registered alongside
    # this kernel.
    @register_kernel(
        "quantization",
        "mxfp8",
        name="flashinfer_quantize_mxfp8",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            required_features=frozenset({"tensor_core:block_scale"}),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={},
        priority=Priority.PERFORMANT,
    )
    def flashinfer_quantize_mxfp8(
        x: torch.Tensor,
        enable_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return mxfp8_quantize(x, False)


# Gate on the FP4 tensor-core capability, not on an arch-version predicate.
# ``is_blackwell`` tests ``major == 10`` exactly, so it excluded sm_120 (RTX PRO
# Blackwell) even though sm_120 has FP4 tensor cores -- ``fp4_quantize`` stayed
# bound to ``error_fn`` and the first nvfp4 forward raised, while the matching
# nvfp4 GEMMs (``cublaslt_mm_nvfp4``, ``flashinfer_mm_nvfp4``) had already
# passed their own gates. ``has_fp4_tensor_cores`` is unchanged on sm_89/sm_90
# (no ``tensor_core:f4``) and on sm_100/sm_103 (has it, as ``is_blackwell`` did).
if platform.is_nvidia and platform.has_fp4_tensor_cores:
    from flashinfer import (
        fp4_quantize,
        nvfp4_block_scale_interleave,
    )

    @register_kernel(
        "quantization",
        "nvfp4",
        name="flashinfer_quantize_nvfp4",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            required_features=frozenset({"tensor_core:f4"}),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={
            "has_scale": frozenset({True}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_quantize_nvfp4(
        x: torch.Tensor,
        scale: float | torch.Tensor | None = None,
        scale_layout: str = "swizzled",
        enable_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The public quantization API uses the actual scale; FlashInfer's FP4
        # helper expects the inverse scale used before packing.
        scale_inv = 1.0 / scale
        return fp4_quantize(
            x,
            global_scale=scale_inv,
            sf_vec_size=16,
            is_sf_swizzled_layout=scale_layout == "swizzled",
            enable_pdl=enable_pdl,
        )


__all__ = [
    "fp4_quantize",
    "mxfp8_quantize",
    "nvfp4_block_scale_interleave",
    "fp8_blockscale_quantize_runner_sm90",
]
