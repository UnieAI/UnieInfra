"""Regression tests for shared MXFP4 MoE weight allocation."""

import os
import sys

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.loaders import load_model_weight
from tokenspeed.runtime.layers.moe.weights.mxfp4 import (
    create_mxfp4_weight_pair,
)


def _mxfp4_spec(
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    tp_rank: int = 0,
    tp_size: int = 1,
):
    return MoELayerSpec(
        top_k=2,
        num_experts=num_local_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_rank=0,
        ep_size=1,
    )


class TestMxfp4Weights:
    def test_scale_weights_store_checkpoint_bytes(self):
        layer = nn.Module()
        create_mxfp4_weight_pair(
            _mxfp4_spec(
                num_local_experts=2,
                hidden_size=64,
                intermediate_size=96,
            ),
            layer,
        )

        assert (layer.w13_weight_scale.dtype) == (torch.uint8)
        assert (layer.w2_weight_scale.dtype) == (torch.uint8)

    def test_e8m0_scale_load_preserves_checkpoint_bytes(self):
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            pytest.skip("torch.float8_e8m0fnu is unavailable")

        layer = nn.Module()
        create_mxfp4_weight_pair(
            _mxfp4_spec(
                num_local_experts=1,
                hidden_size=64,
                intermediate_size=64,
            ),
            layer,
        )

        raw_w1_scale = (
            torch.tensor([120, 121], dtype=torch.uint8).repeat(64).reshape(64, 2)
        )
        load_model_weight(
            layer.w13_weight_scale,
            raw_w1_scale.view(e8m0_dtype),
            "w1",
            local_expert_id=0,
            tp_rank=0,
            is_bias=False,
            use_presharded_weights=False,
            do_transpose=False,
        )
        assert torch.equal(layer.w13_weight_scale.data[0, :64], raw_w1_scale)

        raw_w2_scale = (
            torch.tensor([122, 123], dtype=torch.uint8).repeat(64).reshape(64, 2)
        )
        load_model_weight(
            layer.w2_weight_scale,
            raw_w2_scale.view(e8m0_dtype),
            "w2",
            local_expert_id=0,
            tp_rank=0,
            is_bias=False,
            use_presharded_weights=False,
            do_transpose=False,
        )
        assert torch.equal(layer.w2_weight_scale.data[0], raw_w2_scale)

    def test_aiter_loads_checkpoint_shards_into_padded_buffers(self):
        # DeepSeek-V4.1-Flash at TP4 has 576 intermediate columns per rank;
        # AITER pads them to 640. Rank 1 of 2 must still read checkpoint rows
        # [576, 1152) while writing into views of the padded buffers.
        ispp, padded, hidden, tp_size = 576, 640, 64, 2
        layer = nn.Module()
        create_mxfp4_weight_pair(
            _mxfp4_spec(1, hidden, ispp * tp_size, tp_rank=1, tp_size=tp_size),
            layer,
            solution="aiter",
        )
        w13, w2 = layer.w13_weight, layer.w2_weight
        assert w13.shape == (1, 2 * ispp, hidden // 2)
        assert w13.untyped_storage().nbytes() == 2 * padded * hidden // 2
        assert w2.shape == (1, hidden, ispp // 2)
        assert w2.untyped_storage().nbytes() == hidden * padded // 2
        # w2 scale columns: 640 / 32 = 20, padded to a multiple of 8.
        assert layer.w2_weight_scale.untyped_storage().nbytes() == hidden * 24

        gate = torch.randint(0, 256, (ispp * tp_size, hidden // 2), dtype=torch.uint8)
        up = torch.randint(0, 256, (ispp * tp_size, hidden // 2), dtype=torch.uint8)
        down = torch.randint(0, 256, (hidden, ispp * tp_size // 2), dtype=torch.uint8)
        for shard_id, loaded in (("w1", gate), ("w3", up)):
            load_model_weight(
                w13,
                loaded,
                shard_id,
                local_expert_id=0,
                tp_rank=1,
                is_bias=False,
                use_presharded_weights=False,
                do_transpose=False,
                tp_size=tp_size,
            )
        load_model_weight(
            w2,
            down,
            "w2",
            local_expert_id=0,
            tp_rank=1,
            is_bias=False,
            use_presharded_weights=False,
            do_transpose=False,
            tp_size=tp_size,
        )

        assert torch.equal(w13.data[0, :ispp], gate[ispp:])
        assert torch.equal(w13.data[0, ispp:], up[ispp:])
        assert torch.equal(w2.data[0], down[:, ispp // 2 :])
        buffer = torch.empty(0, dtype=torch.uint8).set_(
            w13.untyped_storage(), 0, (1, 2 * padded, hidden // 2)
        )
        assert not buffer[0, 2 * ispp :].any()

    def test_other_solutions_keep_unpadded_contiguous_buffers(self):
        layer = nn.Module()
        create_mxfp4_weight_pair(_mxfp4_spec(2, 64, 576), layer, solution="gluon")
        for param in layer.parameters():
            assert param.is_contiguous()
            assert param.untyped_storage().nbytes() == param.numel()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
