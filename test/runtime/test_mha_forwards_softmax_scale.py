"""Every MHA kernel call must carry the layer's softmax scale.

`backends/mha.py` called `mha_extend_with_kvcache` and `mha_decode_with_kvcache`
without `softmax_scale`, and each of those kernels defaults a missing scale to
`1 / sqrt(head_dim)`. That is right for every model in this tree whose
`self.scaling` is `head_dim**-0.5` -- and wrong for the ones where it is not:

    models/gemma.py:337    self.scaling = 1.0            (gemma-4; matches HF
                                                          modeling_gemma4.py:921,
                                                          q/k are unit-RMS after
                                                          qk-norm)
    gemma-3-27b            query_pre_attn_scalar 168 against head_dim 128
    models/inkling.py:467  self.scaling = 1.0 / head_dim (not the square root)

On gemma-4 that scaled the logits by 1/16 on the 50 sliding layers (head_dim
256) and 1/22.6 on the 10 global ones (512). A softmax over logits that small
is close to uniform, so attention returns roughly the mean of V, and the engine
answered "What is the capital of France?" with '額LLLLLLL額額額額...' while every
latency number in the harness looked healthy. `backends/inkling.py`, which
wraps this same backend, passed the scale at all three of its call sites the
whole time.

This is checked against the source rather than by running a forward, because
the defect is an omitted keyword: a numerical test would need the exact model
whose scale differs to fail at all, and would pass for Qwen forever.
"""

from __future__ import annotations

import ast
import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="runtime-1gpu")

_MHA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
    "tokenspeed",
    "runtime",
    "layers",
    "attention",
    "backends",
    "mha.py",
)
_KERNELS = {"mha_extend_with_kvcache", "mha_decode_with_kvcache"}


def _kernel_calls(path):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name in _KERNELS:
            yield name, node


class MhaForwardsSoftmaxScaleTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.exists(_MHA), _MHA)
        self.calls = list(_kernel_calls(_MHA))

    def test_the_file_still_calls_both_kernels(self):
        """Guards the guard: if the calls move or get renamed, the assertions
        below would pass vacuously on an empty list."""
        found = {name for name, _ in self.calls}
        self.assertEqual(found, _KERNELS, f"found {found}")

    def test_every_call_passes_softmax_scale(self):
        for name, call in self.calls:
            kwargs = {kw.arg for kw in call.keywords if kw.arg}
            self.assertIn(
                "softmax_scale",
                kwargs,
                f"{name}() at line {call.lineno} does not pass softmax_scale; "
                f"the kernel will fall back to 1/sqrt(head_dim)",
            )

    def test_the_scale_comes_from_the_layer(self):
        """`softmax_scale=None` would satisfy the test above and change
        nothing."""
        for name, call in self.calls:
            for kw in call.keywords:
                if kw.arg != "softmax_scale":
                    continue
                self.assertEqual(
                    ast.unparse(kw.value),
                    "layer.scaling",
                    f"{name}() at line {call.lineno} passes "
                    f"{ast.unparse(kw.value)!r}, not the layer's own scale",
                )


if __name__ == "__main__":
    unittest.main()
