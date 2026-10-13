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

"""Which architectures the trtllm-gen FlashInfer MHA kernels admit."""

import dataclasses

import pytest
from tokenspeed_kernel.platform import ArchVersion
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels


def test_trtllm_gen_extend_is_not_admitted_on_consumer_blackwell(h100_platform):
    load_builtin_kernels()
    spec = KernelRegistry.get().get_by_name("flashinfer_trtllm_mha_extend_with_kvcache")
    if spec is None:
        pytest.skip("registered only on NVIDIA Hopper and newer")

    def on(major: int, minor: int):
        return dataclasses.replace(
            h100_platform, arch_version=ArchVersion(major, minor)
        )

    assert spec.capability.satisfied_by(on(10, 0))
    assert spec.capability.satisfied_by(on(10, 3))
    assert not spec.capability.satisfied_by(on(12, 0))
    assert not spec.capability.satisfied_by(on(12, 1))
