# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Copyright (c) 2026 UnieAI
#
# Ported from SGLang, near-literally:
#   sglang/srt/mem_cache/mamba_checkpoint_pool.py  (Int8CheckpointStore,
#                                                   MambaCheckpointPool)
#   sglang/srt/mem_cache/allocator/mamba.py        (MambaSlotAllocator)
# at pinned commit b61a069.
#
#   Copyright 2023-2026 SGLang Team
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""int8 store for *cached* linear-attention recurrent states.

The point is the separation, not the compression. This engine files a retained
snapshot in the same pool that live requests take their working and checkpoint
slots from, so retaining several snapshots per chunk turned into
``SchedulePrefillEvent: failed to allocate Mamba checkpoint slot`` at 64
concurrent requests, and the fixes that followed only arbitrated a contention
that does not have to exist. SGLang does not have that failure mode because a
cached state lives *here* -- a separate pool, quantized -- and never competes
with a running sequence.

Per cached slot:
  * the SSM temporal state in **int8**, symmetric per (head, k-channel)
  * the conv1d window at its native dtype (W-1 tokens; too small to be worth
    quantizing)

Why int8 rather than fp8: a cached checkpoint is loaded once on a hit and
decoding continues at full precision, so the only error is a single rounding of
S -- it never re-enters the recurrence as a quantize/dequantize loop. The
temporal state is roughly uniform, so int8 with a per-(head, k-channel) scale
beats fp8-e4m3 at the same byte, which spends bits on an exponent. The scale
axis reduces over d_v, matching the per-k-channel decay ``diag(alpha)``, so the
large entries keep roughly bf16 precision and the error lands on small entries
that barely move the readout.

Deviations from the source, both because the surrounding engine differs rather
than because a different design was wanted:

* The factory is not ported. SGLang's reads its global ``get_exec().mamba``
  context; ours will be built from ``ServerArgs`` at the call site.
* ``MambaSlotAllocator`` is ported inline rather than imported, because this
  engine's active slots are owned by the C++ ``MambaChunkAllocator`` and only
  the checkpoint pool needs a Python-side free list. Slot 0 stays reserved so
  the index space matches the source's.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class MambaSlotAllocator:
    """Free list of checkpoint slot indices.

    Request-level rather than per-token, so this deliberately does not inherit
    the KV page allocator. Slot 0 is reserved as a dummy write target for padded
    tokens, which is why ``clear`` starts at 1 and the pools below allocate
    ``num_slots + 1`` rows.
    """

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int = 1) -> torch.Tensor | None:
        """``need_size`` slot indices, or None when the pool is short.

        None rather than an exception: these are cache entries, so a short pool
        must degrade reuse depth rather than fail a request.
        """
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor) -> None:
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self) -> None:
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )


class Int8CheckpointStore:
    """int8 store for cached multi-layer linear-attention states.

    Tensors, indexed by a slot handed out by the caller's allocator:

        qdata : [L, num_slots, H, d_v, d_k]  int8         the quantized state
        scale : [L, num_slots, H, 1,   d_k]  scale_dtype  per (layer, slot, head, k)

    A "state" spans all L recurrent layers for one cached point, matching how
    the radix caches one full state per node. The scale reduces over d_v, so
    each (head, k-channel) gets its own.

    ``scale_dtype`` should match the source state's dtype so quantize and
    dequantize use the identical scale; it is not required to be bf16.
    """

    QMAX = 127

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        device: str,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.H = num_heads
        self.d_v = head_v_dim
        self.d_k = head_k_dim
        self.device = device
        self.qdata = torch.empty(
            num_layers,
            num_slots,
            num_heads,
            head_v_dim,
            head_k_dim,
            dtype=torch.int8,
            device=device,
        )
        self.scale = torch.empty(
            num_layers,
            num_slots,
            num_heads,
            1,
            head_k_dim,
            dtype=scale_dtype,
            device=device,
        )

    @classmethod
    def quantize(cls, state: torch.Tensor):
        """``state [..., H, d_v, d_k]`` -> ``(qint8, scale[..., H, 1, d_k])``.

        amax, scale and round are computed in float32 so quantizing a
        low-precision state does not lose precision in the intermediate --
        symmetric with ``dequantize``, which is already float32. The scale is
        rounded to the state's storage dtype *before* the division, so quantize
        and dequantize use the identical scale.
        """
        state_fp32 = state.to(torch.float32)
        amax = state_fp32.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        scale = (amax / cls.QMAX).to(state.dtype)
        q = (
            torch.round(state_fp32 / scale.to(torch.float32))
            .clamp(-cls.QMAX, cls.QMAX)
            .to(torch.int8)
        )
        return q, scale

    @staticmethod
    def dequantize(q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype):
        return (q.to(torch.float32) * scale.to(torch.float32)).to(out_dtype)

    def store(self, slots: torch.Tensor, state: torch.Tensor) -> None:
        """Quantize and write. ``state``: [L, N, H, d_v, d_k], or [L, H, d_v, d_k]
        for a single slot."""
        if state.dim() == 4:
            state = state.unsqueeze(1)
        q, scale = self.quantize(state)
        self.qdata[:, slots] = q
        self.scale[:, slots] = scale.to(self.scale.dtype)

    def load(self, slots: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize the states at ``slots`` -> [L, N, H, d_v, d_k]."""
        return self.dequantize(self.qdata[:, slots], self.scale[:, slots], out_dtype)

    def copy_to_pool(
        self,
        dst_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        """Dequantize checkpoints straight into the active pool tensor -- the
        copy-on-write a cache hit performs. Output dtype follows the destination.
        """
        dst_temporal[:, dst_slots] = self.load(src_slots, dst_temporal.dtype)

    def store_from_pool(
        self,
        src_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        """Quantize from an active pool tensor into checkpoint slots."""
        self.store(dst_slots, src_temporal[:, src_slots])

    def mem_usage_bytes(self) -> int:
        return (
            self.qdata.numel() * self.qdata.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    def bytes_per_slot(self) -> int:
        return self.mem_usage_bytes() // max(1, self.num_slots)


class MambaCheckpointPool:
    """The radix cache's store for retained recurrent states.

    Separate from the active pool by construction: ``alloc`` here cannot take a
    slot a running sequence needs, which is the property the engine's own
    single-pool arrangement lacks.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: list[tuple],
        conv_dtype: torch.dtype,
        device: str,
        temporal_dtype: torch.dtype | None = None,
    ):
        self.num_slots = num_slots
        self.device = device
        self.temporal = Int8CheckpointStore(
            num_layers=num_layers,
            num_slots=num_slots + 1,  # slot 0 reserved, as in MambaSlotAllocator
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            device=device,
            scale_dtype=(
                temporal_dtype if temporal_dtype is not None else torch.bfloat16
            ),
        )
        # Conv windows stay at their native dtype: one buffer per conv tensor.
        self.conv = [
            torch.empty(
                (num_layers, num_slots + 1) + tuple(shape),
                dtype=conv_dtype,
                device=device,
            )
            for shape in conv_shapes
        ]
        self.allocator = MambaSlotAllocator(size=num_slots, device=device)

    def alloc(self, n: int = 1):
        return self.allocator.alloc(n)

    def free(self, slots: torch.Tensor) -> None:
        self.allocator.free(slots)

    def available_size(self) -> int:
        return self.allocator.available_size()

    def clear(self) -> None:
        """Release every checkpoint slot. The int8 data is left as is; slots are
        overwritten on the next store."""
        self.allocator.clear()

    def store_from_active(
        self,
        active_temporal: torch.Tensor,
        active_conv: list[torch.Tensor],
        active_slots: torch.Tensor,
        ckpt_slots: torch.Tensor,
    ) -> None:
        """Quantize temporal and copy conv from the active pool into checkpoints.

        Takes the tensors rather than SGLang's ``active_mamba_pool`` wrapper --
        this engine reaches its state through the layer pool's
        ``get_mamba_params(layer_id)``, so there is no equivalent object to pass.
        """
        self.temporal.store_from_pool(active_temporal, active_slots, ckpt_slots)
        for i, c in enumerate(self.conv):
            c[:, ckpt_slots] = active_conv[i][:, active_slots]

    def load_to_active(
        self,
        active_temporal: torch.Tensor,
        active_conv: list[torch.Tensor],
        ckpt_slots: torch.Tensor,
        active_slots: torch.Tensor,
    ) -> None:
        """Dequantize temporal and copy conv back into the active pool."""
        self.temporal.copy_to_pool(active_temporal, ckpt_slots, active_slots)
        for i, c in enumerate(self.conv):
            active_conv[i][:, active_slots] = c[:, ckpt_slots].to(active_conv[i].dtype)

    @staticmethod
    def estimate_mem_usage_bytes(
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: list[tuple],
        conv_dtype: torch.dtype,
        temporal_dtype: torch.dtype,
    ) -> dict:
        """Footprint in bytes *without* allocating, so a caller can check it
        against free memory first and fail with an actionable message rather
        than a mid-allocation CUDA OOM."""
        slots = num_slots + 1  # slot 0 reserved
        scale_isz = torch.empty((), dtype=temporal_dtype).element_size()
        conv_isz = torch.empty((), dtype=conv_dtype).element_size()
        qdata = num_layers * slots * num_heads * head_v_dim * head_k_dim  # int8
        scale = num_layers * slots * num_heads * head_k_dim * scale_isz
        conv = 0
        for shape in conv_shapes:
            n = 1
            for s in shape:
                n *= int(s)
            conv += num_layers * slots * n * conv_isz
        return {
            "qdata": qdata,
            "scale": scale,
            "conv": conv,
            "total": qdata + scale + conv,
        }

    def mem_usage_bytes(self) -> int:
        conv_bytes = sum(c.numel() * c.element_size() for c in self.conv)
        return self.temporal.mem_usage_bytes() + conv_bytes
