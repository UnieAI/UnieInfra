# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The fused GDN chunk prefill against the in-tree Triton kernel.

The in-tree GDN tests pin `solution` to triton or flashinfer, so nothing else
calls this kernel with known inputs. It fuses the state and output scans of the
same algorithm, so it is held to the Triton result at bf16 tolerance, on the
fused path (>= FUSED_MIN_SEQS sequences) and on the single-sequence delegate,
and it must be what selection picks for the model's geometry on sm_12x.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12,
    reason="requires an sm_12x device (sm_120 or sm_121)",
)

# Qwen3.5/3.8: 16 query/key heads, 48 value heads, both head dims 128 (TP1).
HQ, HV, DK, DV = 16, 48, 128, 128


def _inputs(seqlens: list[int]):
    torch.manual_seed(0)
    total = sum(seqlens)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn(1, total, HQ, DK, **kw) * 0.1
    k = torch.randn(1, total, HQ, DK, **kw) * 0.1
    v = torch.randn(1, total, HV, DV, **kw) * 0.1
    g = -torch.rand(1, total, HV, device="cuda", dtype=torch.float32) * 0.1
    beta = torch.rand(1, total, HV, device="cuda", dtype=torch.float32)
    state = torch.randn(len(seqlens), HV, DV, DK, device="cuda") * 0.01
    cu = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    return q, k, v, g, beta, state, cu


def _run(solution, q, k, v, g, beta, state, cu):
    from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill

    return gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=DK**-0.5,
        initial_state=state.clone(),
        cu_seqlens=cu,
        qk_l2norm=True,
        output_final_state=True,
        solution=solution,
    )


@pytest.mark.parametrize(
    "seqlens",
    [[1024, 300, 77, 513], [64], [2000]],
    ids=["fused-4seq", "delegate-1seq-short", "delegate-1seq-long"],
)
def test_fused_gdn_prefill_matches_triton(seqlens):
    from tokenspeed_kernel.plugins import discover_plugins

    discover_plugins()
    args = _inputs(seqlens)
    ours = _run("unieinfra_fused", *args)
    ref = _run("triton", *args)
    torch.testing.assert_close(ours.out.float(), ref.out.float(), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(ours.final_state, ref.final_state, rtol=2e-2, atol=2e-3)


def test_selection_picks_the_fused_gdn_prefill():
    from tokenspeed_kernel.ops.attention.gdn import _chunk_prefill_traits
    from tokenspeed_kernel.plugins import discover_plugins
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    discover_plugins()
    fmt = dense_tensor_format(torch.bfloat16)
    kernel = select_kernel(
        "attention",
        "gdn_chunk_prefill",
        format_signature(q=fmt, k=fmt, v=fmt),
        traits=_chunk_prefill_traits(DK, DV, HQ, HV, False, True),
    )
    assert kernel.name == "unieinfra_gdn_chunk_prefill_sm120"
