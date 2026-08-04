"""--mamba-track-interval: the retained-snapshot stride for GDN state pages.

The state group's page size IS the snapshot stride N. These tests pin the
three things that make that safe:

1. N is always a positive multiple of the attention page P (the C++
   coordinator folds group keys on the gcd grid and asserts divisibility).
2. The published state group is sized and captured at N, so it pins ~T/N
   block ids instead of ~T/P.
3. The capacity model is bytes-exact against plan_component_tensors, so the
   logged "x capacity" claim is arithmetic, not a guess.

They also pin the NEGATIVE result: under the flat layout every block id
carries a state row in every state slab, so shrinking the attention page to
a FlashInfer-legal size costs ~5x capacity.
"""

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


def _stub_ceil_div_module() -> None:
    """paged_cache_spec is direct-loaded without package context (its own
    tests do the same), but compute_paged_cache_group_page_counts local-imports
    ``tokenspeed.runtime.utils.common.ceil_div`` -- a module that pulls torch
    and tokenspeed_kernel. Stub the one symbol so the sizing tests stay
    hermetic and torch-free, like the module they exercise.
    """
    import types

    for name in (
        "tokenspeed",
        "tokenspeed.runtime",
        "tokenspeed.runtime.utils",
        "tokenspeed.runtime.utils.common",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    common = sys.modules["tokenspeed.runtime.utils.common"]
    if not hasattr(common, "ceil_div"):
        common.ceil_div = lambda a, b: -(-a // b)


_stub_ceil_div_module()


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_fmp = _load("flat_memory_plan_track_interval", "flat_memory_plan.py")
_pcs = _load("paged_cache_spec_track_interval", "paged_cache_spec.py")

state_snapshot_interval = _fmp.state_snapshot_interval
capacity_model = _fmp.capacity_model
components_from_layers = _fmp.components_from_layers
equalized_block_size = _fmp.equalized_block_size
plan_component_tensors = _fmp.plan_component_tensors
state_const_bytes = _fmp.state_const_bytes

state_group_page_sizes = _pcs.state_group_page_sizes
state_page_size = _pcs.state_page_size
layer_page_sizes = _pcs.layer_page_sizes
group_specs_from_layer_types = _pcs.group_specs_from_layer_types
compute_paged_cache_group_page_counts = _pcs.compute_paged_cache_group_page_counts
compute_max_logical_pages_for_capture = _pcs.compute_max_logical_pages_for_capture


class _DT:
    def __init__(self, itemsize):
        self.itemsize = itemsize


_BF16, _FP32 = _DT(2), _DT(4)

# Qwen3.5-27B, verbatim from the HF config: 64 layers on a full_attention_interval
# of 4 (16 full + 48 GDN), head_dim 256, 4 KV heads, 48 linear value heads.
_QWEN35_27B_LAYERS = tuple(
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
)
_QWEN35_27B_KV_BYTES_PER_SLOT = 2 * 4 * 256 * 2  # K+V, 4 kv heads, head_dim 256, bf16
_QWEN35_27B_STATE_BYTES = state_const_bytes(
    (2 * 16 * 128 + 48 * 128, 4 - 1), _BF16,  # conv column x (kernel - 1)
    (48, 128, 128), _FP32,  # ssm [Hv, V, K]
)


def _qwen35_27b_components():
    return components_from_layers(
        layer_types=list(_QWEN35_27B_LAYERS),
        kv_bytes_per_slot=_QWEN35_27B_KV_BYTES_PER_SLOT,
        state_const_bytes=_QWEN35_27B_STATE_BYTES,
    )


class SnapshotIntervalTest(unittest.TestCase):
    def test_default_256_is_a_no_op_against_the_equalized_page(self):
        # The shipped default (256) is BELOW the equalized page, so today's
        # geometry is reproduced exactly -- the flag cannot regress anyone.
        self.assertEqual(
            state_snapshot_interval(block_size=832, requested_interval=256), 832
        )

    def test_none_and_zero_mean_one_snapshot_per_page(self):
        self.assertEqual(
            state_snapshot_interval(block_size=832, requested_interval=None), 832
        )
        self.assertEqual(
            state_snapshot_interval(block_size=832, requested_interval=0), 832
        )

    def test_rounds_up_to_a_multiple_of_the_attention_page(self):
        # The coordinator asserts group block_size % gcd == 0; rounding up is
        # what keeps gcd(P, N) == P.
        self.assertEqual(
            state_snapshot_interval(block_size=832, requested_interval=1000), 1664
        )
        self.assertEqual(
            state_snapshot_interval(block_size=832, requested_interval=1664), 1664
        )
        for requested in (833, 1663, 1664, 4096, 8192):
            n = state_snapshot_interval(block_size=832, requested_interval=requested)
            self.assertEqual(n % 832, 0, requested)
            self.assertGreaterEqual(n, requested)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            state_snapshot_interval(block_size=0, requested_interval=256)
        with self.assertRaises(ValueError):
            state_snapshot_interval(block_size=832, requested_interval=-1)


class StateGroupPageSizesTest(unittest.TestCase):
    def test_empty_when_the_interval_equals_the_page(self):
        self.assertEqual(
            state_group_page_sizes(
                layer_types=_QWEN35_27B_LAYERS, page_size=832, state_page_size=832
            ),
            {},
        )
        self.assertEqual(
            state_group_page_sizes(
                layer_types=_QWEN35_27B_LAYERS, page_size=832, state_page_size=None
            ),
            {},
        )

    def test_only_state_labels_get_the_coarser_grid(self):
        sizes = state_group_page_sizes(
            layer_types=_QWEN35_27B_LAYERS, page_size=832, state_page_size=6656
        )
        self.assertEqual(sizes, {"linear_attention": 6656})

    def test_rejects_a_non_multiple(self):
        with self.assertRaises(ValueError):
            state_group_page_sizes(
                layer_types=_QWEN35_27B_LAYERS, page_size=832, state_page_size=1000
            )

    def test_pure_attention_model_gets_nothing(self):
        self.assertEqual(
            state_group_page_sizes(
                layer_types=("full_attention",) * 4, page_size=64, state_page_size=512
            ),
            {},
        )


class PublishedGeometryTest(unittest.TestCase):
    def _specs(self, state_page):
        return group_specs_from_layer_types(
            layer_types=_QWEN35_27B_LAYERS,
            sliding_window_tokens=None,
            page_size=832,
            page_sizes=state_group_page_sizes(
                layer_types=_QWEN35_27B_LAYERS,
                page_size=832,
                state_page_size=state_page,
            )
            or None,
        )

    def test_state_group_carries_the_stride_and_attention_keeps_the_page(self):
        specs = {s.group_id: s for s in self._specs(6656)}
        self.assertEqual(specs["full_attention"].rows_per_page, 832)
        self.assertEqual(specs["full_attention"].block_size, 832)
        self.assertEqual(specs["linear_attention"].rows_per_page, 6656)
        self.assertEqual(specs["linear_attention"].block_size, 6656)
        self.assertEqual(specs["linear_attention"].family, "state")

    def test_state_page_size_accessor_reads_the_stride(self):
        self.assertEqual(state_page_size(self._specs(6656), default=832), 6656)
        self.assertEqual(state_page_size(self._specs(832), default=832), 832)
        # No state group published -> caller's default.
        self.assertEqual(state_page_size((), default=832), 832)

    def test_state_group_pins_fewer_pages(self):
        kwargs = dict(
            max_live_requests=32,
            max_scheduled_tokens=8192,
            max_total_tokens=135616,
            max_context_len=262144,
        )
        before = compute_paged_cache_group_page_counts(self._specs(832), **kwargs)
        after = compute_paged_cache_group_page_counts(self._specs(6656), **kwargs)
        self.assertEqual(before["full_attention"], after["full_attention"])
        self.assertLess(after["linear_attention"], before["linear_attention"])
        # min(2 live pages/request + floor(T/N) snapshots,
        #     ceil(T/N) + one partial tail page/request), both + dummy.
        self.assertEqual(
            after["linear_attention"],
            min(32 * 2 + 135616 // 6656, -(-135616 // 6656) + 32) + 1,
        )

    def test_capture_table_narrows_with_the_stride(self):
        state832 = next(s for s in self._specs(832) if s.family == "state")
        state6656 = next(s for s in self._specs(6656) if s.family == "state")
        wide = compute_max_logical_pages_for_capture(state832, max_context_len=262144)
        narrow = compute_max_logical_pages_for_capture(
            state6656, max_context_len=262144
        )
        self.assertEqual(wide, -(-(262144 + 1) // 832))
        self.assertEqual(narrow, -(-(262144 + 1) // 6656))
        self.assertLess(narrow, wide)

    def test_layer_page_sizes_are_per_layer_geometry(self):
        sizes = layer_page_sizes(
            layer_types=_QWEN35_27B_LAYERS,
            sliding_window_tokens=None,
            page_size=832,
            page_sizes={"linear_attention": 6656},
        )
        self.assertEqual(len(sizes), len(_QWEN35_27B_LAYERS))
        for label, size in zip(_QWEN35_27B_LAYERS, sizes):
            self.assertEqual(size, 6656 if label == "linear_attention" else 832)


class CapacityModelTest(unittest.TestCase):
    def test_block_bytes_match_the_planner(self):
        comps = _qwen35_27b_components()
        p = equalized_block_size(
            layer_types=list(_QWEN35_27B_LAYERS),
            kv_bytes_per_slot=_QWEN35_27B_KV_BYTES_PER_SLOT,
            state_const_bytes=_QWEN35_27B_STATE_BYTES,
            block_size=64,
        )
        self.assertEqual(p, 832)
        cap = capacity_model(comps, block_size=p)
        plan = plan_component_tensors(comps, block_size=p, budget_bytes=1 << 40)
        self.assertEqual(cap.block_bytes, plan.geometry.block_bytes)
        self.assertEqual(cap.num_blocks_for(1 << 40), plan.geometry.num_blocks)
        # The reported split: 74% of a page is GDN state, 26% attention KV.
        self.assertEqual(cap.block_bytes, 208470016)
        self.assertAlmostEqual(
            cap.state_bytes_per_block / cap.block_bytes, 0.738, places=3
        )

    def test_a_coarser_stride_buys_capacity_monotonically(self):
        comps = _qwen35_27b_components()
        budget = 163 * 208470016
        base = capacity_model(comps, block_size=832)
        self.assertEqual(base.num_blocks_for(budget), 163)
        self.assertEqual(base.num_blocks_for(budget) * 832, 135616)
        previous = base.tokens_for(budget)
        for n in (1664, 3328, 6656, 13312):
            cap = capacity_model(comps, block_size=832, state_interval=n)
            # The byte plan is untouched: same pages, same nominal tokens.
            self.assertEqual(cap.num_blocks_for(budget), 163)
            self.assertGreater(cap.tokens_for(budget), previous)
            previous = cap.tokens_for(budget)
        # Ceiling: 2x, when snapshots cost nothing (N -> infinity the state
        # term vanishes and only the attention page remains).
        huge = capacity_model(comps, block_size=832, state_interval=832 * 4096)
        self.assertLess(huge.tokens_for(budget), 2 * base.tokens_for(budget))

    def test_shrinking_the_attention_page_is_a_large_regression(self):
        # NEGATIVE RESULT, pinned so nobody re-derives it optimistically:
        # every block id carries a state row in every state slab, so an
        # attention page inside FlashInfer's paged-decode whitelist
        # ({16,32,64,128}) multiplies the state cost per token by 832/P'.
        comps = _qwen35_27b_components()
        budget = 163 * 208470016
        base = capacity_model(comps, block_size=832).tokens_for(budget)
        for flashinfer_page in (16, 32, 64, 128):
            cap = capacity_model(comps, block_size=flashinfer_page)
            self.assertLess(cap.tokens_for(budget), base / 4)
        # Even the coarsest snapshot stride does not rescue P' = 128.
        rescued = capacity_model(comps, block_size=128, state_interval=128 * 4096)
        self.assertLess(rescued.tokens_for(budget), base)


if __name__ == "__main__":
    unittest.main()
