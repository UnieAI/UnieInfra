# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
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

"""XQA decode declares the head_dim it can serve, so selection can route.

flashinfer's XQA compiles head_dim as a template parameter and rejects
anything that is not a multiple of 16 in [16, 256]. Leaving that undeclared
meant selection could hand it a wider layer and the failure arrived from
inside JIT compilation -- `ValueError: Invalid head_dim: 512` -- at CUDA graph
capture, which takes the server down at startup rather than falling back.

Gemma 4 is the shape that needs this: 50 sliding layers at 256 and 10 full
layers at 512 in one model. With the trait declared, selection routes per
layer and only the wide layers fall through to a portable kernel. vLLM
resolves the same conflict one level up, forcing every layer onto Triton for
the whole model.
"""

from __future__ import annotations

import pytest

registry = pytest.importorskip("tokenspeed_kernel.registry")

XQA_DECODE = "flashinfer_trtllm_mha_decode_with_kvcache"


def _spec():
    registry.load_builtin_kernels()
    for spec in registry.KernelRegistry.get().list_kernels(
        "attention", "mha_decode_with_kvcache"
    ):
        if spec.name == XQA_DECODE:
            return spec
    pytest.skip(f"{XQA_DECODE} not registered on this platform")


def test_head_dim_trait_is_declared():
    assert "head_dim" in _spec().traits, (
        "undeclared means selection can pick XQA for a width it cannot compile, "
        "and the error arrives from inside JIT at graph capture"
    )


@pytest.mark.parametrize("head_dim", [16, 64, 128, 256])
def test_supported_widths_are_accepted(head_dim):
    assert head_dim in _spec().traits["head_dim"]


@pytest.mark.parametrize("head_dim", [512, 320, 257, 8, 24])
def test_unsupported_widths_are_refused(head_dim):
    # 512 is Gemma 4's full-attention width, the one that took the server down;
    # 320 is a multiple of 16 above the ceiling; 257 and 24 are not multiples
    # of 16; 8 is below the floor.
    assert head_dim not in _spec().traits["head_dim"]


def test_the_declared_set_is_the_rule_not_a_transcription():
    """Every multiple of 16 in [16, 256] and nothing else."""
    assert set(_spec().traits["head_dim"]) == set(range(16, 257, 16))
