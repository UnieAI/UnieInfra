"""Coverage for the five commits made 2026-08-21. Pure-python, no CUDA."""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path("/workspace/roy/UnieInfra/python")


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


spec_mod = load(
    "spec_leaf", "tokenspeed/runtime/layers/attention/kv_cache/recipes/spec.py"
)
for n in [
    "tokenspeed",
    "tokenspeed.runtime",
    "tokenspeed.runtime.layers",
    "tokenspeed.runtime.layers.attention",
    "tokenspeed.runtime.layers.attention.kv_cache",
    "tokenspeed.runtime.layers.attention.kv_cache.recipes",
]:
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["tokenspeed.runtime.layers.attention.kv_cache.recipes.spec"] = spec_mod
pcs = load("pcs_leaf", "tokenspeed/runtime/configs/paged_cache_spec.py")
split = pcs.split_groups_to_equal_depth


def test_state_labels_are_never_split():  # b62da5a9
    qwen27 = ("full_attention",) * 16 + ("linear_attention",) * 48
    assert set(split(qwen27)) == {"full_attention", "linear_attention"}
    qwen35 = ("full_attention",) * 12 + ("linear_attention",) * 36
    assert set(split(qwen35)) == {"full_attention", "linear_attention"}


def test_sliding_still_splits():  # b62da5a9 must not regress gemma
    g4 = ("sliding_attention",) * 50 + ("full_attention",) * 10
    out = split(g4)
    assert set(out) == {f"sliding_attention_{k}" for k in range(5)} | {"full_attention"}
    from collections import Counter

    assert set(Counter(out).values()) == {10}
    ink = ("sliding_attention",) * 55 + ("full_attention",) * 11
    assert len({t for t in split(ink) if t.startswith("sliding")}) == 5


def test_equal_depth_stack_is_a_noop():
    one_to_one = ("full_attention", "sliding_attention") * 12
    assert split(one_to_one) == one_to_one


def test_numbered_subgroup_resolves_for_any_label():  # de7c4282
    r = spec_mod._retention_for_label
    assert r("linear_attention_0") == r("linear_attention")
    assert r("sliding_attention_2") == r("sliding_attention")
    assert r("full_attention_1") == r("full_attention")
    assert r("nonsense_0") is None
    assert r("trailing_") is None
