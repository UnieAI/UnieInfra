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

"""Separate store for *cached* linear-attention recurrent states.

The point is the separation, not the compression. This engine used to file a
retained snapshot in the same pool that live requests take their working and
checkpoint slots from, so retaining several snapshots per chunk turned into
``SchedulePrefillEvent: failed to allocate Mamba checkpoint slot`` at 64
concurrent requests, and the fixes that followed only arbitrated a contention
that does not have to exist. SGLang does not have that failure mode because a
cached state lives *here* -- a separate pool -- and never competes with a
running sequence.

Separation buys a second thing SGLang does not use it for, and it is the reason
this pool is on by default: **the snapshot no longer has to share the working
state's dtype.** The working state is bf16 because the active pool is sized in
bytes and its dtype sets how many sequences can run at once. A snapshot is not
sized that way -- there are far fewer of them -- and bf16 is exactly wrong for
one, because a resumed prefill starts from it while a cold prefill carries the
same state in an fp32 accumulator and never rounds it. Measured on a GDN hybrid:
a bf16 round trip moves the state by 2.3e-3 relative, the first generated
token's logprob by 0.04-0.07, and at 2048 characters of prompt it flips the
argmax, so the same prompt returns different text depending on whether it hit
the cache. At fp32 the resumed output is bit-identical to the cold one.

Per cached slot:
  * the SSM temporal state at ``ckpt_dtype`` -- float32 by default (exact), or
    **int8** (symmetric per (head, k-channel)) for four times the slots
  * the conv1d window at its native dtype (W-1 tokens; too small to be worth
    either widening or quantizing, and exact there already)

Why int8 rather than fp8 when compression is wanted: a cached checkpoint is
loaded once on a hit and decoding continues at full precision, so the only error
is a single rounding of S -- it never re-enters the recurrence as a
quantize/dequantize loop. The temporal state is roughly uniform, so int8 with a
per-(head, k-channel) scale beats fp8-e4m3 at the same byte, which spends bits
on an exponent. The scale axis reduces over d_v, matching the per-k-channel
decay ``diag(alpha)``, so the large entries keep roughly bf16 precision and the
error lands on small entries that barely move the readout. It is still a
lossy choice, and it is the capacity trade, not the default.

Deviations from the source, both because the surrounding engine differs rather
than because a different design was wanted:

* The factory is not ported. SGLang's reads its global ``get_exec().mamba``
  context; ours is built from ``ServerArgs`` at the call site.
* ``MambaSlotAllocator`` is ported inline rather than imported, because this
  engine's active slots are owned by the C++ ``MambaChunkAllocator`` and only
  the checkpoint pool needs a Python-side free list. Slot 0 stays reserved so
  the index space matches the source's.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Names accepted by ``--mamba-ckpt-dtype``, in the order a help string should
# list them: exact first, then the capacity trades.
CKPT_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}


def resolve_ckpt_dtype(name: str | torch.dtype) -> torch.dtype:
    """``"float32"`` -> ``torch.float32``; a dtype passes through."""
    if isinstance(name, torch.dtype):
        return name
    try:
        return CKPT_DTYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown mamba checkpoint dtype {name!r}; "
            f"expected one of {sorted(CKPT_DTYPES)}"
        ) from None


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
    is_quantized = True

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

    def store_layer(self, layer: int, slots: torch.Tensor, state: torch.Tensor) -> None:
        """Quantize one layer's states into ``slots``. ``state``: [N, H, d_v, d_k].

        Not in the source, which stores all L layers at once because its radix
        cache runs outside the model. Here the publish happens inside a layer's
        forward, where only that layer's slice of the active pool is in hand.
        """
        q, scale = self.quantize(state)
        self.qdata[layer, slots] = q
        self.scale[layer, slots] = scale.to(self.scale.dtype)

    def load_layer(
        self, layer: int, slots: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        """Dequantize one layer's states at ``slots`` -> [N, H, d_v, d_k]."""
        return self.dequantize(
            self.qdata[layer, slots], self.scale[layer, slots], out_dtype
        )

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


class DenseCheckpointStore:
    """Unquantized store for cached multi-layer linear-attention states.

    One tensor, ``data : [L, num_slots, H, d_v, d_k]`` at ``dtype``, indexed by
    a slot handed out by the caller's allocator. The interface is
    ``Int8CheckpointStore``'s so the pool above can hold either without knowing
    which; the difference is that at ``torch.float32`` this one is *exact*, and
    exactness is the whole reason a cached snapshot is stored apart from the
    working state instead of inside it.
    """

    is_quantized = False

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.H = num_heads
        self.d_v = head_v_dim
        self.d_k = head_k_dim
        self.device = device
        self.dtype = dtype
        self.data = torch.empty(
            num_layers,
            num_slots,
            num_heads,
            head_v_dim,
            head_k_dim,
            dtype=dtype,
            device=device,
        )

    def store(self, slots: torch.Tensor, state: torch.Tensor) -> None:
        """``state``: [L, N, H, d_v, d_k], or [L, H, d_v, d_k] for one slot."""
        if state.dim() == 4:
            state = state.unsqueeze(1)
        self.data[:, slots] = state.to(self.dtype)

    def load(self, slots: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return self.data[:, slots].to(out_dtype)

    def store_layer(self, layer: int, slots: torch.Tensor, state: torch.Tensor) -> None:
        self.data[layer, slots] = state.to(self.dtype)

    def load_layer(
        self, layer: int, slots: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        return self.data[layer, slots].to(out_dtype)

    def copy_to_pool(
        self,
        dst_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        dst_temporal[:, dst_slots] = self.load(src_slots, dst_temporal.dtype)

    def store_from_pool(
        self,
        src_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        self.store(dst_slots, src_temporal[:, src_slots])

    def mem_usage_bytes(self) -> int:
        return self.data.numel() * self.data.element_size()

    def bytes_per_slot(self) -> int:
        return self.mem_usage_bytes() // max(1, self.num_slots)


class MambaCheckpointPool:
    """The radix cache's store for retained recurrent states.

    Separate from the active pool by construction, in two ways that matter:
    ``alloc`` here cannot take a slot a running sequence needs, and the
    temporal state is stored at ``ckpt_dtype`` rather than at the active pool's
    ``ssm_dtype``. The default is float32 -- see the module docstring for what
    bf16 costs a cache hit.
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
        ckpt_dtype: torch.dtype | str = torch.float32,
    ):
        """
        Args:
            temporal_dtype: The *active* pool's SSM dtype. Only int8 uses it,
                as the scale dtype, so quantize and dequantize share a scale.
            ckpt_dtype: Storage dtype for the cached temporal state.
                ``torch.int8`` selects the quantized store; anything else the
                dense one.
        """
        ckpt_dtype = resolve_ckpt_dtype(ckpt_dtype)
        self.num_slots = num_slots
        self.device = device
        self.ckpt_dtype = ckpt_dtype
        if ckpt_dtype is torch.int8:
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
        else:
            self.temporal = DenseCheckpointStore(
                num_layers=num_layers,
                num_slots=num_slots + 1,  # slot 0 reserved
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                device=device,
                dtype=ckpt_dtype,
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
        ckpt_dtype: torch.dtype | str = torch.float32,
    ) -> dict:
        """Footprint in bytes *without* allocating, so a caller can check it
        against free memory first and fail with an actionable message rather
        than a mid-allocation CUDA OOM.

        ``qdata``/``scale`` are the int8 store's two tensors; the dense store
        reports its single tensor as ``qdata`` and ``scale`` 0, so a caller can
        log the same three fields either way.
        """
        ckpt_dtype = resolve_ckpt_dtype(ckpt_dtype)
        slots = num_slots + 1  # slot 0 reserved
        conv_isz = torch.empty((), dtype=conv_dtype).element_size()
        state_elems = num_layers * slots * num_heads * head_v_dim * head_k_dim
        if ckpt_dtype is torch.int8:
            scale_isz = torch.empty((), dtype=temporal_dtype).element_size()
            qdata = state_elems  # int8
            scale = num_layers * slots * num_heads * head_k_dim * scale_isz
        else:
            qdata = state_elems * torch.empty((), dtype=ckpt_dtype).element_size()
            scale = 0
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

    @staticmethod
    def bytes_per_slot_estimate(**kwargs) -> int:
        """``estimate_mem_usage_bytes`` divided by the slots it was asked for.

        Sizing code needs one slot's cost before it knows how many to ask for,
        which is the wrong way round for the estimator above.
        """
        kwargs = {**kwargs, "num_slots": 1}
        est = MambaCheckpointPool.estimate_mem_usage_bytes(**kwargs)
        return est["total"] // 2  # `num_slots=1` allocates 2 rows (slot 0)

    def store_layer_from_active(
        self,
        layer: int,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
        active_slots: torch.Tensor,
        ckpt_slots: torch.Tensor,
    ) -> None:
        """One layer's publish: quantize temporal, copy conv verbatim.

        ``ssm_state``/``conv_state`` are this layer's slice of the active pool,
        which is what a layer forward has -- the whole-pool variants above take
        the [L, ...] tensors the radix side would use.
        """
        self.temporal.store_layer(layer, ckpt_slots, ssm_state[active_slots])
        self.conv[0][layer, ckpt_slots] = conv_state[active_slots]

    def load_layer_to_active(
        self,
        layer: int,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
        ckpt_slots: torch.Tensor,
        active_slots: torch.Tensor,
    ) -> None:
        """One layer's cache hit: dequantize temporal, copy conv back."""
        ssm_state[active_slots] = self.temporal.load_layer(
            layer, ckpt_slots, ssm_state.dtype
        )
        conv_state[active_slots] = self.conv[0][layer, ckpt_slots].to(conv_state.dtype)

    def mem_usage_bytes(self) -> int:
        conv_bytes = sum(c.numel() * c.element_size() for c in self.conv)
        return self.temporal.mem_usage_bytes() + conv_bytes

    def bytes_per_slot(self) -> int:
        return self.mem_usage_bytes() // max(1, self.num_slots + 1)

    # -- The two paths the engine's step actually takes ---------------------
    #
    # Both take whole-pool tensors and index vectors that may carry -1, and
    # both must convert rather than alias: the active pool is bf16 and this one
    # is not. Writing `ckpt[dst] = active[src]` with mismatched dtypes would
    # either raise or, with a view in the way, reinterpret bytes.

    def store_from_active_indexed(
        self,
        active_conv: torch.Tensor,
        active_temporal: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        cache_lengths: torch.Tensor | None = None,
        page_size: int = 0,
    ) -> None:
        """Publish working slots ``src`` into checkpoint slots ``dst``.

        ``src`` indexes the active pool, ``dst`` indexes this one -- two index
        spaces, so ``src == dst`` is an ordinary copy here and must not be
        skipped as a self-copy. Entries out of range (the -1 the scheduler
        writes when it had no slot to give) are dropped, and when
        ``page_size > 0`` so are rows whose ``cache_lengths`` is not on a page
        boundary.
        """
        from tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton import (  # noqa: E501
            fused_mamba_state_copy,
        )

        fused_mamba_state_copy(
            active_conv,
            src_indices,
            dst_indices,
            cache_lengths=cache_lengths,
            page_size=page_size,
            dst_pool=self.conv[0],
        )
        if not self.temporal.is_quantized:
            fused_mamba_state_copy(
                active_temporal,
                src_indices,
                dst_indices,
                cache_lengths=cache_lengths,
                page_size=page_size,
                dst_pool=self.temporal.data,
            )
            return

        # Quantizing needs the values on the Python side, so this arm gathers
        # per layer. Invalid rows are steered to slot 0 -- the reserved dummy --
        # rather than masked out, because masking would need `.any()` and this
        # runs inside the overlap-scheduled decode step, where a device sync
        # costs more than a wasted write.
        keep = self._keep_mask(
            src_indices, dst_indices, active_temporal.shape[1], cache_lengths, page_size
        )
        src = torch.where(keep, src_indices.long(), 0)
        dst = torch.where(keep, dst_indices.long(), 0)
        for layer in range(self.temporal.num_layers):
            self.temporal.store_layer(layer, dst, active_temporal[layer][src])

    def load_to_active_indexed(
        self,
        active_conv: torch.Tensor,
        active_temporal: torch.Tensor,
        ckpt_indices: torch.Tensor,
        active_indices: torch.Tensor,
    ) -> None:
        """Seed working slots ``active_indices`` from checkpoints ``ckpt_indices``.

        The copy-on-write a cache hit performs. Rows whose source is -1 keep
        whatever the working slot already held.
        """
        from tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton import (  # noqa: E501
            fused_mamba_state_copy,
        )

        fused_mamba_state_copy(
            self.conv[0],
            ckpt_indices,
            active_indices,
            dst_pool=active_conv,
        )
        if not self.temporal.is_quantized:
            fused_mamba_state_copy(
                self.temporal.data,
                ckpt_indices,
                active_indices,
                dst_pool=active_temporal,
            )
            return

        keep = self._keep_mask(
            ckpt_indices, active_indices, self.temporal.num_slots, None, 0
        )
        if not bool(keep.any()):
            return
        src = ckpt_indices.long()[keep]
        dst = active_indices.long()[keep]
        for layer in range(self.temporal.num_layers):
            active_temporal[layer][dst] = self.temporal.load_layer(
                layer, src, active_temporal.dtype
            )

    @staticmethod
    def _keep_mask(
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        src_pool_size: int,
        cache_lengths: torch.Tensor | None,
        page_size: int,
    ) -> torch.Tensor:
        keep = (src_indices >= 0) & (dst_indices >= 0) & (src_indices < src_pool_size)
        if page_size > 0 and cache_lengths is not None:
            keep = keep & (cache_lengths % page_size == 0)
        return keep
