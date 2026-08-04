"""Mixed prefill+decode batching is gated on attention-backend capability.

``--enable-mixed-batch`` makes the C++ scheduler emit plans that hold prefill
chunks and decode rows in the same iteration (``ForwardMode.MIXED``). MSA cannot
describe such a plan -- its extend metadata has no decode tail -- so it declares
``supports_mixed_batch = False`` and rejects MIXED at metadata init. The MHA
family (mha/fa3/fa4/triton/flashinfer) splits the batch at the segment boundary
and IS capable, except on speculative/draft ranks where the decode tail is a
verify window rather than one token per row; those instances declare themselves
incapable.

These tests pin the startup guard (``validate_mixed_batch_backend``, called
from engine/event_loop before the C++ Scheduler ctor) so the flag fails fast at
launch instead of killing the scheduler on the first step that happens to mix.
"""

from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine.event_loop import validate_mixed_batch_backend


class _MixedCapableBackend:
    supports_mixed_batch = True


class _MixedIncapableBackend:
    supports_mixed_batch = False


class _LegacyBackend:
    """No capability attribute at all -- must be treated as capable."""


class MixedBatchBackendGuardTest(unittest.TestCase):
    def test_disabled_flag_never_raises(self):
        # The guard must be inert whenever the feature is off, even on a
        # backend that cannot serve MIXED (this is the shipped default).
        validate_mixed_batch_backend(
            enable_mixed_batch=False,
            attn_backend=_MixedIncapableBackend(),
            backend_name="triton",
        )

    def test_capable_backend_passes(self):
        validate_mixed_batch_backend(
            enable_mixed_batch=True,
            attn_backend=_MixedCapableBackend(),
            backend_name="trtllm",
        )

    def test_backend_without_attribute_defaults_to_capable(self):
        validate_mixed_batch_backend(
            enable_mixed_batch=True,
            attn_backend=_LegacyBackend(),
            backend_name="mla",
        )

    def test_incapable_backend_raises_with_actionable_message(self):
        with self.assertRaises(ValueError) as ctx:
            validate_mixed_batch_backend(
                enable_mixed_batch=True,
                attn_backend=_MixedIncapableBackend(),
                backend_name="triton",
            )
        msg = str(ctx.exception)
        self.assertIn("--enable-mixed-batch", msg)
        self.assertIn("triton", msg)
        self.assertIn("_MixedIncapableBackend", msg)

    def test_auto_selected_backend_names_itself_auto(self):
        with self.assertRaises(ValueError) as ctx:
            validate_mixed_batch_backend(
                enable_mixed_batch=True,
                attn_backend=_MixedIncapableBackend(),
                backend_name=None,
            )
        self.assertIn("auto", str(ctx.exception))


class BackendCapabilityDeclarationTest(unittest.TestCase):
    """The declarations the guard reads must match the backends' own asserts."""

    def test_base_class_defaults_to_capable(self):
        from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend

        self.assertTrue(AttentionBackend.supports_mixed_batch)

    def test_mha_family_declares_capable(self):
        from tokenspeed.runtime.layers.attention.backends.mha import MHAAttnBackend

        self.assertTrue(MHAAttnBackend.supports_mixed_batch)

    def test_mha_instance_opts_out_under_speculation(self):
        # The class-level True is the plain-decode case; a spec/draft instance
        # must turn it back off in __init__ (via mixed_batch_supported) so the
        # startup guard -- not a mid-serve NotImplementedError -- rejects the
        # flag. __init__ assigns exactly this call's result.
        from tokenspeed.runtime.layers.attention.backends.mha import (
            mixed_batch_supported,
        )

        self.assertTrue(mixed_batch_supported(1, False, False))
        self.assertFalse(mixed_batch_supported(4, False, False))
        self.assertFalse(mixed_batch_supported(1, True, False))
        self.assertFalse(mixed_batch_supported(1, False, True))

    def test_msa_declares_incapable(self):
        from tokenspeed.runtime.layers.attention.backends.msa import MSAAttnBackend

        self.assertFalse(MSAAttnBackend.supports_mixed_batch)


if __name__ == "__main__":
    unittest.main()
