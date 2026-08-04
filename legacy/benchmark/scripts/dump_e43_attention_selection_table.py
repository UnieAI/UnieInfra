# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the UnieInfra project
"""E43 evidence: the attention decision table, per simulated arch.

Prints, for every simulated compute capability, every request shape the layer
was asked about, what UnieInfra's capability registry chose, and what upstream's
hardcoded selector chose for the identical request. This is the artefact that
turns "behaviour is unchanged" from a claim into a table.

Run inside the ``unieinfra-vllm-build`` container::

    CUDA_VISIBLE_DEVICES=1 python benchmark/scripts/dump_e43_attention_selection_table.py
"""

from __future__ import annotations

import itertools
import json
import sys

import torch
import vllm.v1.attention.unieinfra_selection as uas
from unieinfra_kernel.boundary import ArchVersion, KernelRegistry, Platform
from unieinfra_kernel.selection.platform import PlatformInfo, _get_cuda_sm_features
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform as vllm_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import AttentionSelectorConfig

ARCHS = [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]


def _pin(major: int, minor: int) -> DeviceCapability:
    arch = ArchVersion(major, minor)
    Platform.override(
        PlatformInfo(
            vendor="nvidia",
            arch_version=arch,
            device_name=f"simulated sm_{major}{minor}",
            device_count=1,
            total_memory=96 * 1024**3,
            memory_bandwidth=1500.0,
            sm_count=128,
            max_threads_per_sm=2048,
            max_shared_memory_per_sm=100 * 1024,
            sm_features=_get_cuda_sm_features(arch),
            runtime_features=frozenset({"runtime:cuda_graph"}),
        )
    )
    capability = DeviceCapability(major, minor)
    type(vllm_platform).get_device_capability = classmethod(  # type: ignore[assignment]
        lambda cls, device_id=0: capability
    )
    KernelRegistry.reset()
    uas._registered = False
    uas._registered_names.clear()
    return capability


def _shapes():
    """Yield (label, config, num_heads) for the shapes worth showing."""
    for dtype in (torch.bfloat16, torch.float16):
        for kv in ("auto", "fp8"):
            for swa in (False, True):
                for sink in (False, True):
                    yield (
                        f"dense hs=128 {str(dtype).split('.')[-1]} kv={kv} "
                        f"swa={int(swa)} sink={int(sink)}",
                        AttentionSelectorConfig(
                            head_size=128,
                            dtype=dtype,
                            kv_cache_dtype=kv,
                            block_size=None,
                            has_sink=sink,
                            has_sliding_window=swa,
                            attn_type=AttentionType.DECODER,
                        ),
                        32,
                    )
    yield (
        "dense hs=128 bf16 non-causal",
        AttentionSelectorConfig(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            block_size=None,
            use_non_causal=True,
        ),
        32,
    )
    for sparse in (False, True):
        for kv in ("auto", "fp8_ds_mla"):
            for num_heads in (16, 128):
                yield (
                    f"MLA hs=576 sparse={int(sparse)} kv={kv} heads={num_heads}",
                    AttentionSelectorConfig(
                        head_size=576,
                        dtype=torch.bfloat16,
                        kv_cache_dtype=kv,
                        block_size=None,
                        use_mla=True,
                        use_sparse=sparse,
                    ),
                    num_heads,
                )


def main() -> int:
    rows = []
    mismatches = 0
    with set_current_vllm_config(VllmConfig()):
        for major, minor in ARCHS:
            capability = _pin(major, minor)
            for label, config, num_heads in _shapes():
                try:
                    want = vllm_platform.get_attn_backend_cls(
                        None, attn_selector_config=config, num_heads=num_heads
                    )
                except ValueError as exc:
                    want = f"<none: {str(exc).split('.')[0]}>"
                got = uas.resolve_attention_backend_cls(
                    None, attn_selector_config=config, num_heads=num_heads
                )
                agree = got is None or got == want
                mismatches += 0 if agree else 1
                rows.append(
                    {
                        "arch": f"sm_{major}{minor}",
                        "shape": label,
                        "upstream": (want or "").rsplit(".", 1)[-1],
                        "unieinfra": (got or "DECLINED -> upstream").rsplit(".", 1)[-1],
                        "agree": agree,
                    }
                )
            del capability

    width = max(len(r["shape"]) for r in rows)
    current = None
    for row in rows:
        if row["arch"] != current:
            current = row["arch"]
            print(f"\n=== {current} " + "=" * 40)
            print(f"{'shape'.ljust(width)}  {'upstream':<28} {'unieinfra':<28} ok")
        print(
            f"{row['shape'].ljust(width)}  {row['upstream']:<28} "
            f"{row['unieinfra']:<28} {'y' if row['agree'] else 'N'}"
        )

    print(f"\nrows={len(rows)} mismatches={mismatches}")
    print("counters:", json.dumps(uas._counters))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
