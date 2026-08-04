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

"""Architecture gating: capability predicates must not be version comparisons.

``ArchVersion`` only compares version numbers, so an open-ended
``min_arch_version`` such as ``(9, 0)`` or ``(10, 0)`` is satisfied by *every*
newer part -- including sm_120 (RTX PRO / workstation Blackwell, arch 12.0),
which lacks capabilities that every datacenter part from sm_100 on has. These
tests pin the capability semantics so a version predicate cannot silently be
substituted back in.

Ground truth used here, verified with ``ptxas`` from CUDA 12.9:

* ``tcgen05.alloc`` assembles for ``sm_100a`` and is rejected for both
  ``sm_90a`` and ``sm_120a`` -> tcgen05/TMEM is datacenter-Blackwell only.
* ``mapa.shared::cluster`` assembles for ``sm_90a``, ``sm_100a`` **and**
  ``sm_120a``, and the driver reports
  ``CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH == 1`` on sm_120 -> clusters and
  distributed shared memory are *not* what sm_120 is missing.
"""

from __future__ import annotations

import pytest
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    InterconnectInfo,
    PlatformInfo,
    _get_cuda_sm_features,
    current_platform,
)
from tokenspeed_kernel.registry import KernelRegistry

# --- synthetic platforms ------------------------------------------------------
#
# Built through the real `_get_cuda_sm_features` so the feature set under test is
# the one the detector actually produces, not a hand-copied duplicate of it.


def _nvidia_platform(major: int, minor: int, smem: int, name: str) -> PlatformInfo:
    arch_version = ArchVersion(major, minor)
    return PlatformInfo(
        vendor="nvidia",
        arch_version=arch_version,
        device_name=name,
        device_count=1,
        total_memory=80 * (1024**3),
        memory_bandwidth=3350.0,
        sm_count=132,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=smem,
        sm_features=_get_cuda_sm_features(arch_version),
        runtime_features=frozenset({"runtime:cuda_graph"}),
        interconnect=InterconnectInfo(topology="single_gpu"),
    )


# sm_86 / sm_89 are the workstation Ampere+Ada parts in the compat matrix;
# sm_90 is H100/H200; sm_100 is B200; sm_103 is B300 -- NEWER than the sm_100
# most gates were written against; sm_120 is the RTX PRO 6000.
SM86 = _nvidia_platform(8, 6, 101376, "NVIDIA RTX A6000")
SM89 = _nvidia_platform(8, 9, 101376, "NVIDIA L40S")
SM90 = _nvidia_platform(9, 0, 232448, "NVIDIA H200")
SM100 = _nvidia_platform(10, 0, 232448, "NVIDIA B200")
SM103 = _nvidia_platform(10, 3, 232448, "NVIDIA B300")
SM120 = _nvidia_platform(12, 0, 101376, "NVIDIA RTX PRO 6000 Blackwell")

ALL_PLATFORMS = {
    "sm_86": SM86,
    "sm_89": SM89,
    "sm_90": SM90,
    "sm_100": SM100,
    "sm_103": SM103,
    "sm_120": SM120,
}


# --- capability features ------------------------------------------------------


@pytest.mark.parametrize(
    "platform,expected",
    [
        pytest.param(SM86, False, id="sm_86"),
        pytest.param(SM89, False, id="sm_89"),
        pytest.param(SM90, False, id="sm_90"),
        pytest.param(SM100, True, id="sm_100"),
        pytest.param(SM103, True, id="sm_103-b300"),
        # The whole point: sm_120 is arch 12.0, so it satisfies every
        # `>= (10, 0)` comparison, yet it has no TMEM.
        pytest.param(SM120, False, id="sm_120"),
    ],
)
def test_tcgen05_is_datacenter_blackwell_only(
    platform: PlatformInfo, expected: bool
) -> None:
    assert platform.has_tcgen05 is expected
    assert ("compute:tcgen05" in platform.sm_features) is expected


@pytest.mark.parametrize(
    "platform,expected",
    [
        pytest.param(SM86, False, id="sm_86"),
        pytest.param(SM89, False, id="sm_89"),
        pytest.param(SM90, False, id="sm_90"),
        pytest.param(SM100, True, id="sm_100"),
        pytest.param(SM103, True, id="sm_103-b300"),
        # sm_120 *does* have FP4 tensor cores. `is_blackwell` (major == 10)
        # said otherwise, which is what kept the nvfp4 quantizer unbound.
        pytest.param(SM120, True, id="sm_120"),
    ],
)
def test_fp4_tensor_cores_include_sm120(platform: PlatformInfo, expected: bool) -> None:
    assert platform.has_fp4_tensor_cores is expected


def test_fp4_capability_disagrees_with_is_blackwell_only_on_sm120() -> None:
    """`has_fp4_tensor_cores` must be a strict superset of `is_blackwell`.

    This is the safety property behind the nvfp4 gate change: swapping
    ``is_blackwell`` for ``has_fp4_tensor_cores`` may only *add* sm_120, never
    change any datacenter or Ampere-class answer.
    """
    for name, platform in ALL_PLATFORMS.items():
        if name == "sm_120":
            assert platform.is_blackwell is False
            assert platform.has_fp4_tensor_cores is True
        else:
            assert platform.has_fp4_tensor_cores is platform.is_blackwell, name


def test_cluster_support_is_not_what_sm120_lacks() -> None:
    """sm_120 has thread-block clusters and DSMEM (ptxas + driver verified).

    Recorded so nobody "fixes" sm_120 by taking ``compute:cluster`` away from
    it -- that would exclude kernels which genuinely do run there.
    """
    assert "compute:cluster" in SM120.sm_features
    assert "memory:tma" in SM120.sm_features
    assert "compute:cluster" not in SM89.sm_features


# --- the gates that were wrong ------------------------------------------------
#
# Each op below is registered with an open-ended `min_arch_version` that sm_120
# satisfied. Reverting the fix (dropping `required_features`) makes the sm_120
# assertion fail while leaving the datacenter assertions passing -- which is
# exactly the bug shape being guarded against.

TCGEN05_GATED_OPS = [
    ("attention", "dsa_decode", "flashinfer_trtllm_dsa_decode"),
    ("attention", "dsa_prefill", "flashinfer_trtllm_dsa_prefill"),
    ("moe", "apply", "flashinfer_trtllm_fp8_moe_apply"),
    ("moe", "apply", "flashinfer_cutedsl_deepep_nvfp4_moe_apply"),
]


def _spec(family: str, mode: str, name: str):
    for spec in KernelRegistry.get().list_kernels(family=family, mode=mode):
        if spec.name == name:
            return spec
    pytest.skip(f"{name} is not registered on this build/platform")


@pytest.mark.parametrize(
    "family,mode,name", TCGEN05_GATED_OPS, ids=[o[2] for o in TCGEN05_GATED_OPS]
)
def test_tcgen05_kernels_are_excluded_on_sm120_and_kept_on_datacenter(
    family: str, mode: str, name: str
) -> None:
    capability = _spec(family, mode, name).capability

    # sm_120 must be filtered out: these wrap tcgen05 cubins (TRT-LLM-gen /
    # CuTe DSL sm100). Every one of them has a portable Triton sibling for the
    # same op, so excluding them leaves a working path rather than a
    # NoKernelFoundError.
    assert capability.satisfied_by(SM120) is False
    assert "compute:tcgen05" in capability.missing_features(SM120)

    # ... and datacenter behaviour is unchanged. B300 (sm_103) matters as much
    # as B200 here: it is newer than the sm_100 these gates were written for.
    assert capability.satisfied_by(SM100) is True
    assert capability.satisfied_by(SM103) is True

    # Pre-Blackwell parts were already excluded by min_arch_version and stay so.
    assert capability.satisfied_by(SM90) is False
    assert capability.satisfied_by(SM86) is False


def test_open_ended_min_arch_alone_is_now_unrepresentable() -> None:
    """The bug shape can no longer be written at all.

    `min_arch_version=(10, 0)` with no upper bound is what every mis-gated
    registration used to carry: it admits sm_120 *and* would admit any future
    arch. E30 fixed the instances; this pins the *form*, so the class cannot
    come back.
    """
    with pytest.raises(ValueError) as excinfo:
        CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        )
    message = str(excinfo.value)
    # The message must say what to add, not merely that something is wrong.
    assert "required_features" in message
    assert "max_arch_version" in message

    # Restating it as the capability it stands for fixes sm_120 without
    # touching sm_100/sm_103.
    capability_gated = CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        required_features=frozenset({"compute:tcgen05"}),
        vendors=frozenset({"nvidia"}),
    )
    assert capability_gated.satisfied_by(SM120) is False
    assert capability_gated.satisfied_by(SM100) is True
    assert capability_gated.satisfied_by(SM103) is True


# --- nvfp4 unblock ------------------------------------------------------------


def test_nvfp4_quantizer_is_registered_wherever_fp4_tensor_cores_exist() -> None:
    """The nvfp4 quantizer and the nvfp4 GEMM must agree about the device.

    ``fp4_quantize`` used to hide behind ``is_blackwell`` (``major == 10``
    exactly) while ``cublaslt_mm_nvfp4`` gated on ``min_arch_version=(10, 0)``.
    On sm_120 that left the registry with an nvfp4 *GEMM* and no nvfp4
    *quantizer*, so the first nvfp4 forward raised.
    """
    platform = current_platform()
    if not (platform.is_nvidia and platform.has_fp4_tensor_cores):
        pytest.skip("device has no FP4 tensor cores")

    from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
    from tokenspeed_kernel.registry import error_fn

    assert fp4_quantize is not error_fn

    quantizers = {
        spec.name
        for spec in KernelRegistry.get().list_kernels(
            family="quantization", mode="nvfp4"
        )
    }
    assert "flashinfer_quantize_nvfp4" in quantizers

    gemms = {
        spec.name
        for spec in KernelRegistry.get().list_kernels(family="gemm", mode="mm")
    }
    assert gemms & {"cublaslt_mm_nvfp4", "flashinfer_mm_nvfp4"}
