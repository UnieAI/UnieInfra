import unittest

import torch

from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.utils.env import global_server_args_dict


def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
    )


def _single_rank_mapping() -> Mapping:
    mapping = Mapping(rank=0, world_size=1)
    global_server_args_dict["mapping"] = mapping
    return mapping


class TestQwen3DenseWeightLoader(unittest.TestCase):
    """ModelOpt checkpoints carry KV-cache scales the dense loader must skip.

    An NVFP4 (or FP8-KV) ModelOpt checkpoint declares
    ``kv_cache_quant_algo`` and ships ``self_attn.k_proj.k_scale`` /
    ``self_attn.v_proj.v_scale`` alongside the projections. Those are only
    materialised as parameters when the KV cache is actually quantised, so with
    ``kv_cache_dtype=auto`` there is no matching entry in ``params_dict``.

    Before the fix, ``k_proj.k_scale`` was rewritten to ``qkv_proj.k_scale`` by
    the stacked-params mapping and then looked up unconditionally, so loading
    ``nvidia/Qwen3-8B-NVFP4`` died with
    ``KeyError: 'model.layers.35.self_attn.qkv_proj.k_scale'`` before the first
    forward. ``qwen3_moe.py`` already skipped these; the dense path did not.
    """

    def test_load_weights_skips_absent_kv_cache_scales(self):
        from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM

        model = Qwen3ForCausalLM(
            _tiny_qwen3_config(),
            mapping=_single_rank_mapping(),
        )
        params = dict(model.named_parameters())
        self.assertNotIn("model.layers.0.self_attn.qkv_proj.k_scale", params)

        # k_scale/v_scale sit on the k/v projections, so they go through the
        # stacked-params branch (k_proj -> qkv_proj); input_scale on o_proj
        # exercises the non-stacked branch.
        model.load_weights(
            [
                (
                    "model.layers.0.self_attn.k_proj.k_scale",
                    torch.tensor(0.5),
                ),
                (
                    "model.layers.0.self_attn.v_proj.v_scale",
                    torch.tensor(0.25),
                ),
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    torch.full((32, 16), 3.0),
                ),
            ]
        )

        # The real weight still loaded; the scales were dropped, not fatal.
        reloaded = dict(model.named_parameters())
        qkv = reloaded["model.layers.0.self_attn.qkv_proj.weight"]
        self.assertEqual(qkv[:32].mean().item(), 3.0)

    def test_load_weights_still_raises_on_a_genuinely_unknown_weight(self):
        """The skip must be narrow: only KV-scale suffixes, only when absent."""
        from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM

        model = Qwen3ForCausalLM(
            _tiny_qwen3_config(),
            mapping=_single_rank_mapping(),
        )
        with self.assertRaises(KeyError):
            model.load_weights(
                [("model.layers.0.self_attn.q_proj.not_a_real_thing", torch.zeros(1))]
            )


if __name__ == "__main__":
    unittest.main()
