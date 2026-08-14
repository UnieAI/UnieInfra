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

"""The DeepGEMM FP8 MoE must be reachable without DeepEP.

``deepep_fp8`` is registered with ``supports_all_to_all_ep`` and raises
"DeepEP MoE plan is missing deepep_group" on a plan that carries no process
group, which left DeepGEMM unusable for single-box tensor parallel -- the
configuration Qwen3.5-122B-A10B-FP8 runs in. These cover the two things that
went wrong on the way to fixing that, both of which produce a working-looking
failure rather than an obvious one.
"""

import pytest
import torch
from tokenspeed_kernel.platform import Platform

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not Platform.get().is_nvidia:
    pytest.skip("NVIDIA-only registration under test", allow_module_level=True)

contiguous_fp8 = pytest.importorskip(
    "tokenspeed_kernel.ops.moe.deep_gemm.contiguous_fp8"
)

pytestmark = pytest.mark.skipif(
    not hasattr(contiguous_fp8, "deep_gemm_contiguous_fp8_moe_apply"),
    reason="DeepGEMM is an optional dependency and is not installed",
)

_BLOCK = 128


def test_counts_are_padded_per_expert_and_leave_idle_experts_empty():
    """Row counts must tile-align, and unrouted experts must cost nothing.

    ``deepep_scatter`` rejects any count that is not a multiple of the block,
    so an unpadded histogram fails outright. Padding every expert instead of
    only the routed ones is the quiet failure: it still runs, but a single
    decode token allocates one block per expert -- 256 x 128 rows for 8 real
    ones -- and only shows up as a throughput mystery.
    """
    num_experts = 8
    # tokens 0..2 all route to experts 0 and 1; nothing routes to 2..7
    topk_ids = torch.tensor(
        [[0, 1], [0, 1], [0, 1]], dtype=torch.int64, device="cuda"
    )
    counts = contiguous_fp8._aligned_expert_counts(topk_ids, num_experts)

    assert len(counts) == num_experts
    assert all(c % _BLOCK == 0 for c in counts), counts
    assert counts[0] == _BLOCK and counts[1] == _BLOCK
    assert counts[2:] == [0] * (num_experts - 2)


def test_counts_grow_by_whole_blocks():
    """A block-and-one-token expert takes two blocks, not one."""
    ids = torch.zeros((_BLOCK + 1, 1), dtype=torch.int64, device="cuda")
    counts = contiguous_fp8._aligned_expert_counts(ids, 2)
    assert counts == [2 * _BLOCK, 0]


def test_registered_without_all_to_all_ep():
    """The trait that makes this reachable where ``deepep_fp8`` is not.

    Both register under solution ``deep_gemm``; ``supports_all_to_all_ep`` is
    what separates them. If this flips to True the selector can hand a
    DeepEP-less plan to a kernel that demands a process group, which is the
    original defect.
    """
    from tokenspeed_kernel.registry import KernelRegistry

    spec = KernelRegistry.get().get_by_name("deep_gemm_contiguous_fp8_moe_apply")
    assert spec is not None, "contiguous DeepGEMM MoE is not registered"
    traits = spec.traits
    assert traits["supports_all_to_all_ep"] == frozenset({False})
    assert traits["supports_ep"] == frozenset({True})
    assert traits["weight_dtype"] == frozenset({"fp8"})
