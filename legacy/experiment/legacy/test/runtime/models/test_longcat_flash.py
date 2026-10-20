"""Cheap LongCat-Flash model wiring tests."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.models.longcat_flash import (
    LongcatFlashForCausalLM,
    _ensure_longcat_config,
    _get_longcat_moe_quant_config,
    _RuntimeLongcatMoE,
)


class TestLongcatFlashRegistry(unittest.TestCase):
    def test_registered(self):
        from tokenspeed.runtime.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["LongcatFlashForCausalLM"])
        self.assertIs(cls, LongcatFlashForCausalLM)
        self.assertEqual(arch, "LongcatFlashForCausalLM")

    def test_mla_and_double_attention_metadata_registered(self):
        from tokenspeed.runtime.configs import model_config

        self.assertIn("LongcatFlashForCausalLM", model_config._MLA_ARCHITECTURES)
        self.assertIn(
            "LongcatFlashForCausalLM",
            model_config._DOUBLE_ATTENTION_LAYER_ARCHITECTURES,
        )


class TestLongcatFlashConfig(unittest.TestCase):
    def test_config_aliases_are_normalized(self):
        config = SimpleNamespace(
            num_layers=28,
            ffn_hidden_size=14336,
            expert_ffn_hidden_size=2048,
            moe_topk=8,
            hidden_size=6144,
            n_routed_experts=512,
        )

        _ensure_longcat_config(config)

        self.assertEqual(config.num_hidden_layers, 28)
        self.assertEqual(config.intermediate_size, 14336)
        self.assertEqual(config.moe_intermediate_size, 2048)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.zero_expert_num, 0)
        self.assertFalse(config.router_bias)


class TestLongcatMixedFp8Config(unittest.TestCase):
    def test_moe_layer_uses_unquantized_backend_when_all_experts_are_ignored(self):
        config = SimpleNamespace(n_routed_experts=2)
        quant_config = SimpleNamespace(
            ignored_layers=[
                f"model.layers.0.mlp.experts.{expert_id}.{proj_name}"
                for expert_id in range(2)
                for proj_name in ("gate_proj", "up_proj", "down_proj")
            ]
        )

        self.assertIsNone(
            _get_longcat_moe_quant_config(
                config,
                quant_config,
                "model.layers.0.mlp",
            )
        )

    def test_moe_layer_keeps_quantization_when_no_experts_are_ignored(self):
        config = SimpleNamespace(n_routed_experts=2)
        quant_config = SimpleNamespace(ignored_layers=[])

        self.assertIs(
            _get_longcat_moe_quant_config(
                config,
                quant_config,
                "model.layers.0.mlp",
            ),
            quant_config,
        )

    def test_moe_layer_rejects_partially_ignored_experts(self):
        config = SimpleNamespace(n_routed_experts=2)
        quant_config = SimpleNamespace(
            ignored_layers=[
                "model.layers.0.mlp.experts.0.gate_proj",
            ]
        )

        with self.assertRaisesRegex(ValueError, "partially ignored"):
            _get_longcat_moe_quant_config(
                config,
                quant_config,
                "model.layers.0.mlp",
            )


class TestLongcatZeroExpert(unittest.TestCase):
    def test_identity_zero_expert_masks_and_adds_hidden_state(self):
        moe = object.__new__(_RuntimeLongcatMoE)
        moe.zero_expert_num = 1
        moe.n_routed_experts = 3
        moe.zero_expert_type = "identity"
        hidden_states = torch.tensor(
            [[2.0, 4.0], [6.0, 8.0]],
            dtype=torch.float32,
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
            topk_ids=torch.tensor([[0, -1], [3, 1]]),
            router_logits=torch.zeros(2, 4),
        )

        zero_output = _RuntimeLongcatMoE._apply_zero_experts(
            moe,
            hidden_states,
            topk_output,
        )

        torch.testing.assert_close(
            zero_output,
            torch.tensor([[1.5, 3.0], [3.0, 4.0]]),
        )
        torch.testing.assert_close(
            topk_output.topk_weights,
            torch.tensor([[0.25, 0.0], [0.0, 0.5]]),
        )
        torch.testing.assert_close(
            topk_output.topk_ids,
            torch.tensor([[0, 0], [0, 1]]),
        )


class TestLongcatCheckpointLoading(unittest.TestCase):
    def test_missing_kv_scale_params_are_silent(self):
        model = object.__new__(LongcatFlashForCausalLM)
        with mock.patch(
            "tokenspeed.runtime.models.longcat_flash._longcat_logger.warning"
        ) as warning:
            self.assertIsNone(model.get_param({}, "model.layers.0.self_attn.0.k_scale"))
            self.assertIsNone(model.get_param({}, "model.layers.0.self_attn.1.v_scale"))
        warning.assert_not_called()

    def test_missing_mtp_params_are_silent(self):
        model = object.__new__(LongcatFlashForCausalLM)
        with mock.patch(
            "tokenspeed.runtime.models.longcat_flash._longcat_logger.warning"
        ) as warning:
            self.assertIsNone(
                model.get_param({}, "model.mtp.layers.0.self_attn.q_proj.weight")
            )
        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
