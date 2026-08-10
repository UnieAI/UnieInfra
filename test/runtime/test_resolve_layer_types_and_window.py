from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "configs"
)


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pcs = _load("paged_cache_spec_layer_type_resolution", "paged_cache_spec.py")
resolve_layer_types_and_window = _pcs.resolve_layer_types_and_window


class _Cfg:
    """Minimal stand-in for a PretrainedConfig: only the attributes we read."""

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


_GEMMA_LABELS = (
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
)


class ResolveLayerTypesAndWindowTest(unittest.TestCase):
    # --- the bug this helper exists to fix -------------------------------

    def test_composite_config_reads_the_text_sub_config(self):
        """Gemma 3/4 shape: top level declares neither field, text_config both."""
        text = _Cfg(layer_types=_GEMMA_LABELS, sliding_window=512)
        top = _Cfg(text_config=text)

        labels, window = resolve_layer_types_and_window(top, text)

        self.assertEqual(labels, _GEMMA_LABELS)
        self.assertEqual(window, 512)

    def test_composite_config_without_fallback_would_collapse(self):
        """Guard the regression directly: top level alone yields the wrong answer.

        A single full-attention group is what the pool falls back to, and
        `validate_paged_cache_group_ids` skips its check when there is only one
        group -- so this misread runs the sliding layers as full attention
        without raising anywhere.
        """
        text = _Cfg(layer_types=_GEMMA_LABELS, sliding_window=512)
        top = _Cfg(text_config=text)

        labels, window = resolve_layer_types_and_window(top, top)

        self.assertEqual(labels, ())
        self.assertIsNone(window)

    # --- no-regression arms: every currently-supported SWA config --------

    def test_flat_config_is_unchanged(self):
        """A flat config passes hf_text_config is hf_config; fallback is a no-op."""
        flat = _Cfg(layer_types=("full_attention", "sliding_attention"), sliding_window=128)

        self.assertEqual(
            resolve_layer_types_and_window(flat, flat),
            resolve_layer_types_and_window(flat, None),
        )
        self.assertEqual(
            resolve_layer_types_and_window(flat, flat),
            (("full_attention", "sliding_attention"), 128),
        )

    def test_forwarding_config_answers_at_the_top_level(self):
        """Qwen3.5 / Kimi K3 / Inkling shape: attribute access forwards down.

        The top level already answers, so the text-config arm is never reached
        even though the two configs disagree.
        """
        text = _Cfg(layer_types=("full_attention",), sliding_window=999)
        top = _Cfg(layer_types=("linear_attention", "full_attention"), sliding_window=64)

        self.assertEqual(
            resolve_layer_types_and_window(top, text),
            (("linear_attention", "full_attention"), 64),
        )

    def test_paged_cache_layer_types_still_wins(self):
        """The engine-declared override outranks transformers' own labels."""
        cfg = _Cfg(
            paged_cache_layer_types=("sliding_attention_0", "full_attention"),
            layer_types=("sliding_attention", "full_attention"),
            sliding_window=32,
        )

        labels, _ = resolve_layer_types_and_window(cfg, cfg)

        self.assertEqual(labels, ("sliding_attention_0", "full_attention"))

    def test_paged_cache_layer_types_on_the_text_config_outranks_its_layer_types(self):
        text = _Cfg(
            paged_cache_layer_types=("sliding_attention_0",),
            layer_types=("sliding_attention",),
        )
        top = _Cfg(text_config=text)

        labels, _ = resolve_layer_types_and_window(top, text)

        self.assertEqual(labels, ("sliding_attention_0",))

    # --- degenerate inputs ------------------------------------------------

    def test_no_labels_anywhere_yields_empty(self):
        cfg = _Cfg()
        self.assertEqual(resolve_layer_types_and_window(cfg, cfg), ((), None))

    def test_none_config_is_tolerated(self):
        self.assertEqual(resolve_layer_types_and_window(None, None), ((), None))

    def test_empty_label_list_falls_through_to_the_text_config(self):
        """An empty list is falsy, so it must not shadow a populated sub-config."""
        text = _Cfg(layer_types=_GEMMA_LABELS, sliding_window=512)
        top = _Cfg(layer_types=[], sliding_window=None, text_config=text)

        self.assertEqual(
            resolve_layer_types_and_window(top, text), (_GEMMA_LABELS, 512)
        )

    def test_window_falls_back_independently_of_labels(self):
        """Labels at the top level, window only below: both must resolve."""
        text = _Cfg(sliding_window=256)
        top = _Cfg(layer_types=("sliding_attention",), text_config=text)

        self.assertEqual(
            resolve_layer_types_and_window(top, text), (("sliding_attention",), 256)
        )

    def test_labels_are_returned_as_a_tuple(self):
        cfg = _Cfg(layer_types=["full_attention", "sliding_attention"])

        labels, _ = resolve_layer_types_and_window(cfg, cfg)

        self.assertIsInstance(labels, tuple)


if __name__ == "__main__":
    unittest.main()
