# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""How does vLLM split KV memory between the full-attention and mamba groups?

Runs vLLM's own grouping and config code on hand-built specs for
Qwen3.6-35B-A3B, so it needs no GPU and no weights. The specs are checked
against the running engine's log first: it reported "attention block size 1056"
and "mamba page size and attention page size are exactly equal", which both
fall out of the arithmetic below.

The question is whether the mamba group's tensor is sized by its actual
per-request need (align mode holds ~2 pages) or by the shared `num_blocks`,
because that decides whether `mamba_align_retained_states` buys snapshots with
attention capacity.
"""

import sys

import torch

sys.path.insert(0, "/workspace/roy/vllm")

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core import kv_cache_utils as ku
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

# From config.json: 40 layers, full_attention_interval 4 -> 10 full / 30 GDN.
N_FULL, N_MAMBA = 10, 30
BLOCK = 1056
# 2 (K+V) * 2 kv heads * 256 head_dim * 2 bytes = 2048 B/token/layer
ATTN_PAGE = BLOCK * 2048
# ssm 32*128*128 fp32 + conv 8192*4 bf16
SSM = (32, 128, 128)
CONV = (8192, 4)

AVAILABLE = int(7.66 * 1024**3)


def build_specs():
    attn = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=2, head_size=256, dtype=torch.bfloat16
    )
    mamba = MambaSpec(
        shapes=(SSM, CONV),
        dtypes=(torch.float32, torch.bfloat16),
        block_size=BLOCK,
        page_size_padded=ATTN_PAGE,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    specs = {}
    for i in range(N_FULL):
        specs[f"attn.{i}"] = attn
    for i in range(N_MAMBA):
        specs[f"gdn.{i}"] = mamba
    return specs, attn, mamba


def main():
    specs, attn, mamba = build_specs()
    print(f"attention page {attn.page_size_bytes:,} B")
    print(f"mamba     page {mamba.page_size_bytes:,} B")
    assert attn.page_size_bytes == mamba.page_size_bytes, "log says these are equal"

    for retained in (0, 4):
        cfg = _cfg(retained)
        groups = ku.get_kv_cache_groups(cfg, specs)
        print(f"\n=== mamba_align_retained_states = {retained} ===")
        print(f"{len(groups)} groups, sizes {[len(g.layer_names) for g in groups]}")
        for g in groups:
            kind = type(g.kv_cache_spec).__name__
            pages = ku.cdiv(
                g.kv_cache_spec.max_memory_usage_bytes(cfg),
                g.kv_cache_spec.page_size_bytes,
            )
            print(f"  {kind:<20} {len(g.layer_names):>3} layers  "
                  f"max_mem {pages:>4} pages/request")

        per_req = ku._max_memory_usage_bytes_from_groups(cfg, groups)
        print(f"  per max-length request: {per_req / 1024**3:.3f} GiB"
              f"  -> concurrency {AVAILABLE / per_req:.2f}x")

        kvc = ku.get_kv_cache_config_from_groups(cfg, groups, AVAILABLE)
        print(f"  num_blocks = {kvc.num_blocks}")
        total = 0
        for t in kvc.kv_cache_tensors:
            total += t.size
        print(f"  tensors: {len(kvc.kv_cache_tensors)}, total {total / 1024**3:.3f} GiB")
        by_kind = {}
        for t in kvc.kv_cache_tensors:
            name = t.shared_by[0]
            kind = "mamba" if name.startswith("gdn") else "attn"
            by_kind[kind] = by_kind.get(kind, 0) + t.size
        for k, v in sorted(by_kind.items()):
            print(f"    {k:<6} {v / 1024**3:7.3f} GiB  ({100 * v / total:.1f}%)")


def _cfg(retained):
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(
        model="/workspace/models/Qwen/Qwen3.6-35B-A3B",
        max_model_len=49152,
        mamba_cache_mode="align",
        mamba_block_size=BLOCK,
        block_size=BLOCK,
        enable_prefix_caching=True,
        mamba_align_retained_states=retained,
    )
    return args.create_engine_config()


if __name__ == "__main__":
    main()
