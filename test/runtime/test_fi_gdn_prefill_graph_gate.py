"""The flashinfer-GDN prefill-graph guard must ask the model, not the flag.

``resolve_fi_gdn_prefill_graph`` exists because the flashinfer GDN
chunk-prefill kernel answers every request with ``"!!!!!!!!"`` under a prefill
capture while reporting better throughput -- a failure no latency gate catches.
Turning the graph off is the right response *for a model that runs that
kernel*.

It was keyed on an environment variable alone, so it also fired for models with
no linear-attention layer at all. Measured on gemma-4-31B-it (50
sliding_attention + 10 full_attention, zero GDN layers) on sm_120 at
concurrency 1: a 1-token-in/1-token-out request cost 96.9 ms against vLLM's
49.7 ms for the same thing, while per-decode-token was at parity (45.5 vs 44.1)
and 8k-token prefill was 6.4% faster here. The whole end-to-end gap at that point
was an intercept, and the intercept was a graph withheld for a kernel the model
never calls.

The discriminating case is gemma vs qwen3.5: if those two ever agree, this file
is asserting nothing.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="runtime-1gpu")


def _model_config(text_layer_types=None, top_layer_types=None):
    """A stand-in for ModelConfig carrying only what the gate reads."""
    text = SimpleNamespace()
    top = SimpleNamespace()
    if text_layer_types is not None:
        text.layer_types = text_layer_types
    if top_layer_types is not None:
        top.layer_types = top_layer_types
    return SimpleNamespace(hf_text_config=text, hf_config=top)


GEMMA4 = ["sliding_attention"] * 50 + ["full_attention"] * 10
QWEN35 = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
KIMI_K3 = ["linear_attention", "attention"]


class ModelMayUseFiGdnTest(unittest.TestCase):
    def setUp(self):
        try:
            from tokenspeed.runtime.engine.event_loop import model_may_use_fi_gdn
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs the runtime package: {exc}")
        self.fn = model_may_use_fi_gdn

    def test_gemma_has_no_linear_layer(self):
        self.assertFalse(self.fn(_model_config(text_layer_types=GEMMA4)))

    def test_qwen35_hybrid_has_one(self):
        self.assertTrue(self.fn(_model_config(text_layer_types=QWEN35)))

    def test_kimi_k3_hybrid_has_one(self):
        self.assertTrue(self.fn(_model_config(text_layer_types=KIMI_K3)))

    def test_gemma_and_qwen_must_disagree(self):
        """Without this the other three can all pass on a constant function."""
        self.assertNotEqual(
            self.fn(_model_config(text_layer_types=GEMMA4)),
            self.fn(_model_config(text_layer_types=QWEN35)),
        )

    def test_layer_types_only_on_the_top_config_is_still_read(self):
        self.assertTrue(self.fn(_model_config(top_layer_types=QWEN35)))
        self.assertFalse(self.fn(_model_config(top_layer_types=GEMMA4)))

    def test_no_layer_types_keeps_todays_behaviour(self):
        """Unknown shape -> conservative. Narrowing the guard for a model whose
        mixers were never read here would trade a correctness guard for a
        speedup on an untested path."""
        self.assertTrue(self.fn(_model_config()))

    def test_none_config_is_conservative(self):
        self.assertTrue(self.fn(None))


class ResolveFiGdnPrefillGraphTest(unittest.TestCase):
    """The gate has to reach `server_args.disable_prefill_graph`, not just
    return the right bool. An earlier version of this resolution printed a
    warning after ModelExecutorConfig had already frozen the bucket ladder and
    changed nothing at all."""

    def setUp(self):
        try:
            from tokenspeed.runtime.engine.event_loop import (
                resolve_fi_gdn_prefill_graph,
            )
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs the runtime package: {exc}")
        self.fn = resolve_fi_gdn_prefill_graph
        self._saved = os.environ.get("TOKENSPEED_ENABLE_FI_GDN_SM120")
        os.environ["TOKENSPEED_ENABLE_FI_GDN_SM120"] = "1"

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("TOKENSPEED_ENABLE_FI_GDN_SM120", None)
        else:
            os.environ["TOKENSPEED_ENABLE_FI_GDN_SM120"] = self._saved

    def test_gemma_keeps_its_prefill_graph(self):
        args = SimpleNamespace(disable_prefill_graph=False)
        self.fn(args, _model_config(text_layer_types=GEMMA4))
        self.assertFalse(args.disable_prefill_graph)

    def test_hybrid_still_loses_it(self):
        args = SimpleNamespace(disable_prefill_graph=False)
        self.fn(args, _model_config(text_layer_types=QWEN35))
        self.assertTrue(args.disable_prefill_graph)

    def test_unknown_model_still_loses_it(self):
        args = SimpleNamespace(disable_prefill_graph=False)
        self.fn(args, _model_config())
        self.assertTrue(args.disable_prefill_graph)

    def test_env_off_keeps_the_graph_for_everyone(self):
        os.environ["TOKENSPEED_ENABLE_FI_GDN_SM120"] = "0"
        args = SimpleNamespace(disable_prefill_graph=False)
        self.fn(args, _model_config(text_layer_types=QWEN35))
        self.assertFalse(args.disable_prefill_graph)

    def test_an_explicit_disable_is_not_undone(self):
        args = SimpleNamespace(disable_prefill_graph=True)
        self.fn(args, _model_config(text_layer_types=GEMMA4))
        self.assertTrue(args.disable_prefill_graph)


if __name__ == "__main__":
    unittest.main()
