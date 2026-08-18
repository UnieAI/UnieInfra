import unittest

from tokenspeed.runtime.configs.qwen3_5_text_base_config import Qwen3_5BaseTextConfig


class OutputGateTypeGuardTest(unittest.TestCase):
    """Qwen3.8 ships `output_gate_type`; Qwen3.5 does not.

    The GDN output gate is hardcoded to silu (`layernorm_gated`:
    ``norm(x) * silu(z)``), so the field is only safe while it names silu.
    Before this guard it landed in ``**kwargs`` and was dropped, which is
    correct for Qwen3.8's "swish" and silently wrong for anything else --
    the checkpoint would serve fluent, plausible, incorrect text, and no
    throughput or latency number would show it.
    """

    def test_default_is_silu_so_qwen3_5_is_unaffected(self):
        self.assertEqual(Qwen3_5BaseTextConfig().output_gate_type, "silu")

    def test_swish_is_accepted_because_swish_is_silu(self):
        # Qwen3.8-27B's value. vLLM folds the two names the same way in
        # `qwen_gdn_linear_attn.py`, so accepting it changes no arithmetic.
        config = Qwen3_5BaseTextConfig(output_gate_type="swish")
        self.assertEqual(config.output_gate_type, "swish")

    def test_an_unimplemented_gate_is_refused_at_construction(self):
        with self.assertRaises(ValueError) as caught:
            Qwen3_5BaseTextConfig(output_gate_type="sigmoid")
        message = str(caught.exception)
        self.assertIn("sigmoid", message)
        # The message has to name the file to change, not just the problem.
        self.assertIn("layernorm_gated", message)


if __name__ == "__main__":
    unittest.main()
