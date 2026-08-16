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

"""The split-reduction decode MoE must match a plain reference, and stay put.

At one token this layer is bandwidth-bound -- ~42 MFLOP against ~50 MB of
weights -- and a grouped-GEMM tiling has no M to parallelise over. Splitting the
reduction adds a third grid axis and measured 1.48x at one token on sm_120,
decaying to parity by 64. Two things have to hold for that to be worth having:
the numbers must be right, and the layout it reads must not be the swapped copy
the CUTLASS path prepares for itself.
"""

import pytest
import torch
from tokenspeed_kernel.platform import Platform

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not Platform.get().is_nvidia:
    pytest.skip("NVIDIA-only kernel under test", allow_module_level=True)

bf16_decode = pytest.importorskip("tokenspeed_kernel.ops.moe.triton.bf16_decode")

pytestmark = pytest.mark.skipif(
    not hasattr(bf16_decode, "decode_moe_body"),
    reason="split-decode MoE is unavailable on this platform",
)

# Qwen3.6-35B-A3B's MoE geometry, shrunk only on the expert axis: the reduction
# blocks index their partials unmasked, so hidden and intermediate have to stay
# multiples of the block width. Shrinking those instead made the kernel read
# uninitialised memory and return a different answer every call -- which is the
# behaviour test_refuses_shapes_it_would_corrupt now pins.
HIDDEN = 512
EXPERTS = 16
TOP_K = 4
INTER = 128


class _Weights(torch.nn.Module):
    """Checkpoint-layout weights: w13 = [gate | up] concatenated, w2 plain."""

    def __init__(self):
        super().__init__()
        self.top_k = TOP_K
        self.w13_weight = torch.nn.Parameter(
            torch.randn(EXPERTS, 2 * INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
            * 0.05,
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(EXPERTS, HIDDEN, INTER, dtype=torch.bfloat16, device="cuda")
            * 0.05,
            requires_grad=False,
        )
        self.ep_size, self.ep_rank, self.tp_size, self.tp_rank = 1, 0, 1, 0


def _reference(x, w, topk_weights, topk_ids):
    """Expert by expert in fp32 -- slow, but leaves nothing to interpret."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(TOP_K):
            expert = int(topk_ids[token, slot])
            hidden = x[token].to(torch.float32) @ w.w13_weight[expert].to(torch.float32).T
            gate, up = hidden[:INTER], hidden[INTER:]
            act = torch.nn.functional.silu(gate) * up
            out[token] += float(topk_weights[token, slot]) * (
                act @ w.w2_weight[expert].to(torch.float32).T
            )
    return out


def _routing(tokens, seed):
    torch.manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    logits = torch.randn(tokens, EXPERTS, dtype=torch.float32, device="cuda")
    scores = torch.softmax(logits, dim=-1)
    weights, ids = torch.topk(scores, k=TOP_K, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.float32)
    return x, weights, ids.to(torch.int32)


@pytest.mark.parametrize("tokens", [1, 2, 8])
def test_matches_fp32_reference(tokens):
    w = _Weights()
    x, weights, ids = _routing(tokens, seed=tokens)
    got = bf16_decode.decode_moe_body(x, w, weights, ids).to(torch.float32)
    ref = _reference(x, w, weights, ids)
    scale = ref.abs().max().clamp(min=1e-6)
    # bf16 inputs against an fp32 reference; the partials are accumulated in
    # fp32, so the error should stay well inside a percent.
    assert ((got - ref).abs().max() / scale).item() < 0.02


def test_reduction_is_bit_reproducible():
    """The docstring's reason for a buffer instead of atomic_add.

    Atomics make the sum order depend on which program finishes first, and a
    forward pass that is not bit-reproducible cannot validate anything
    downstream of it.
    """
    w = _Weights()
    x, weights, ids = _routing(4, seed=7)
    first = bf16_decode.decode_moe_body(x, w, weights, ids)
    for _ in range(4):
        again = bf16_decode.decode_moe_body(x, w, weights, ids)
        assert torch.equal(first, again)


def test_is_not_registered_as_a_solution():
    """Reachable only through the shape-gated delegation, never by selection.

    Registering it as a `moe.apply` solution made the registry choose it for
    every shape, prefill included, and the token gate never ran because
    selection happens once at plan build. End to end on 35B agent C=16 that
    was 49.9 tok/s against the grouped GEMM's 160.1.
    """
    from tokenspeed_kernel.registry import KernelRegistry

    assert (
        KernelRegistry.get().get_by_name("triton_bf16_split_decode_moe_apply") is None
    ), "this kernel must not compete in selection; it is delegated to by shape"
    assert bf16_decode._MAX_DECODE_TOKENS == 16, (
        "the token gate is a measured crossover, not a guess; re-measure before "
        "changing it"
    )


@pytest.mark.parametrize(
    "hidden,inter",
    [
        (512, 64),   # intermediate below the reduction block
        (320, 128),  # hidden not a multiple of the reduction block
    ],
)
def test_refuses_shapes_it_would_corrupt(hidden, inter):
    """A shape the blocks do not divide must raise, not return garbage.

    The partials buffer is uninitialised and the reduction loads are unmasked,
    so an unsupported shape produced a different answer on every call while
    still timing normally. Failing loudly lets the caller fall back to the
    grouped GEMM, which carries no such requirement.

    Divisibility is the whole rule, not a size floor: hidden 384 with
    intermediate 128 clears every block width and was measured reproducible at
    0.29% against the fp32 reference, so it is deliberately not in this list.
    """

    class _Odd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = TOP_K
            self.w13_weight = torch.nn.Parameter(
                torch.zeros(EXPERTS, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda"),
                requires_grad=False,
            )
            self.w2_weight = torch.nn.Parameter(
                torch.zeros(EXPERTS, hidden, inter, dtype=torch.bfloat16, device="cuda"),
                requires_grad=False,
            )

    x = torch.zeros(1, hidden, dtype=torch.bfloat16, device="cuda")
    weights = torch.full((1, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device="cuda")
    ids = torch.zeros(1, TOP_K, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="multiples of"):
        bf16_decode.decode_moe_body(x, _Odd(), weights, ids)


@pytest.mark.parametrize("tokens", [1, 16, 17])
def test_cutlass_hands_over_at_the_measured_crossover(tokens):
    """The grouped-GEMM op routes decode-shaped batches here, and only those.

    Both sides must be right for the handover to be invisible: the split path
    reads w13 in whatever order the CUTLASS weight preprocessor left it, so a
    silently wrong gate/up order would produce silu(up)*gate -- a plausible
    tensor with no error raised anywhere.
    """
    cutlass_unquant = pytest.importorskip(
        "tokenspeed_kernel.ops.moe.flashinfer.cutlass_unquant"
    )
    apply_fn = getattr(cutlass_unquant, "flashinfer_cutlass_unquant_moe_apply", None)
    if apply_fn is None:
        pytest.skip("unquantized CUTLASS MoE is unavailable")

    reference_weights = _Weights()
    engine_weights = _Weights()
    engine_weights.w13_weight.data.copy_(reference_weights.w13_weight.data)
    engine_weights.w2_weight.data.copy_(reference_weights.w2_weight.data)
    for name in dir(cutlass_unquant):
        if name.endswith("_moe_weights"):
            getattr(cutlass_unquant, name)({}, engine_weights)
            break

    x, weights, ids = _routing(tokens, seed=tokens + 100)
    logits = torch.randn(tokens, EXPERTS, dtype=torch.float32, device="cuda")
    got = apply_fn(
        {}, x, engine_weights, logits, topk_weights=weights, topk_ids=ids
    ).to(torch.float32)
    ref = _reference(x, reference_weights, weights, ids)
    scale = ref.abs().max().clamp(min=1e-6)
    assert ((got - ref).abs().max() / scale).item() < 0.02
