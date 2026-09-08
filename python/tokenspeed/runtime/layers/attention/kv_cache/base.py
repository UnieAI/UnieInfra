# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.kvcache.triton import zero_byte_ranges

from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    PagedCacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    PagedCacheGroupSpec,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker

logger = get_colorful_logger(__name__)


def external_field_view(
    buffer: torch.Tensor, plan: CacheMemoryPlan, field_id: str, dtype: torch.dtype
) -> torch.Tensor:
    """One recurrent-state field as a view into the shared buffer's tail.

    Row ``r`` of the returned tensor is slot ``r``: rows ascend with the
    address, and the state pool hands out its highest ids first so the occupied
    rows stay against the end of the buffer and the free ones stay next to the
    attention parents. The stride between rows is the whole slot -- every
    recurrent group's page for one slot sits together -- so a group's rows are
    strided, not contiguous, while each row itself is contiguous, which is what
    the GDN kernels' page ABI requires.

    Args:
        buffer: The unified ``uint8`` arena, ``plan.unified_buffer_bytes`` long.
        plan: Layout whose ``max_state_slots`` is greater than zero.
        field_id: An external field.
        dtype: Element type, which must match the plan's element size.

    Returns:
        A ``(max_state_slots, *field.shape)`` view. No copy.

    Raises:
        ValueError: If the plan has no state region, the field is not
            external, the dtype disagrees with the plan, the group's page count
            disagrees with the slot count, or the buffer is too small.
    """
    if plan.max_state_slots <= 0:
        raise ValueError("plan has no state region; max_state_slots is 0")
    field = plan.field(field_id)
    if not field.external:
        raise ValueError(f"{field_id} is not an external field")
    if torch.empty((), dtype=dtype).element_size() != field.element_size:
        raise ValueError(f"field {field_id!r}: dtype itemsize does not match plan")
    pages = plan.group(field.group_id).page_count
    if pages != plan.max_state_slots:
        # The scheduler addresses these rows by page id. If the group is sized
        # for a different number of pages than the region has slots, every id
        # past the shorter of the two lands somewhere it was not meant to, and
        # a wrong recurrent state is silent.
        raise ValueError(
            f"group {field.group_id!r} has {pages} pages but the state region "
            f"has {plan.max_state_slots} slots"
        )
    if buffer.numel() < plan.unified_buffer_bytes:
        raise ValueError(
            f"buffer holds {buffer.numel()} bytes, the plan needs "
            f"{plan.unified_buffer_bytes}"
        )
    element_strides = []
    stride = 1
    for extent in reversed(field.shape):
        element_strides.append(stride)
        stride *= extent
    offset = plan.state_field_offset_bytes(field_id, 0)
    if offset % field.element_size:
        raise ValueError(
            f"field {field_id!r}: slot base {offset} is not aligned to its element"
        )
    return buffer.view(dtype).as_strided(
        (plan.max_state_slots, *field.shape),
        (plan.state_slot_bytes // field.element_size, *reversed(element_strides)),
        offset // field.element_size,
    )


class CachePool:
    """Own page-backed cache memory and expose backend-specific views."""

    # Pools that alias recurrent-state bytes and KV in one buffer must
    # zero physical pages on reuse to avoid poisoned tails. Pure-attention
    # pools do not alias state, so reused pages need no sanitization.
    paged_cache_requires_page_zeroing: bool = False

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        page_size: int,
        rank: int,
        memory_plan: CacheMemoryPlan,
        *,
        paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...] = (),
        token_capacity: int | None = None,
        backing_pool: CachePool | None = None,
        field_layer_offset: int = 0,
        pd_disaggregation_enabled: bool = False,
    ):
        self.dtype = dtype
        self.rank = rank
        self.size = size
        self.page_size = page_size
        # PD disaggregation transfers the whole LCM arena over the cache-group
        # contract (get_pd_cache_contract): one physical parent page per
        # Mooncake unit, so a strided multi-field arena moves correctly. Every
        # contract-planned pool supports it uniformly.
        self._pd_disaggregation_enabled = bool(pd_disaggregation_enabled)
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            #  Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device
        self.plan = memory_plan
        self._field_layer_offset = int(field_layer_offset)
        if self._field_layer_offset < 0:
            raise ValueError("field_layer_offset must be non-negative")
        # The cache recipe is the single source of the scheduler group specs
        # (CachePoolSpec.paged_cache_group_specs); the pool aligns their
        # physical fields with the memory plan and publishes the runtime
        # contract from the pair. Pools constructed without specs (tests)
        # publish no contract.
        self.runtime_contract: PagedCacheRuntimeContract | None = None
        self.paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...] = ()
        self.paged_cache_group_page_counts: dict[str, int] = {}
        if paged_cache_group_specs:
            self._publish_runtime_contract(
                paged_cache_group_specs,
                token_capacity if token_capacity is not None else size,
            )
        # Allocate lazily when the first field is bound. Concrete pools do
        # that inside their memory-saver region, so the shared buffer keeps
        # the same sleep/wake lifetime as the legacy per-buffer allocations.
        #
        # A heterogeneous draft view (for example, an MHA Eagle3 head over an
        # MLA target) binds its own field family but must not allocate another
        # arena. Construction is deliberately target-first: the draft aliases
        # the target's already-registered buffer and field registry. Sharing
        # the registry is also required by pd_contract(), which validates that
        # every field in the merged plan has acquired a runtime dtype.
        self._backing_pool = backing_pool
        if backing_pool is None:
            self.buffer: torch.Tensor | None = None
            self._fields: dict[str, torch.Tensor] = {}
        else:
            if backing_pool.plan != memory_plan:
                raise ValueError("a cache view must share its backing pool's plan")
            if backing_pool.buffer is None:
                raise ValueError(
                    "the backing cache pool must bind its fields before a view"
                )
            if paged_cache_group_specs:
                raise ValueError(
                    "a cache view must inherit, not republish, the runtime contract"
                )
            self.buffer = backing_pool.buffer
            self._fields = backing_pool._fields
            self.runtime_contract = backing_pool.runtime_contract
            self.paged_cache_group_specs = backing_pool.paged_cache_group_specs
            self.paged_cache_group_page_counts = (
                backing_pool.paged_cache_group_page_counts
            )

        # default state for optional layer-wise transfer control
        self.layerwise_load_tracker = None
        logger.info(
            f"Initialized token to kv pool with size {size}, dtype {dtype}, device {device}, page size {page_size}, rank {rank}"
        )

    def _publish_runtime_contract(
        self,
        group_specs: tuple[PagedCacheGroupSpec, ...],
        token_capacity: int,
    ) -> None:
        """Align recipe group specs with the memory plan and publish the
        scheduler contract. The plan is the source of truth for per-group
        packing and page counts, so every spec group must be planned."""
        from dataclasses import replace

        plan_groups = {group.group_id: group for group in self.plan.groups}
        aligned = []
        counts: dict[str, int] = {}
        for spec in group_specs:
            if spec.group_id in counts:
                raise ValueError(
                    f"cache group {spec.group_id!r} is published more than once"
                )
            group = plan_groups.get(spec.group_id)
            if group is None:
                raise ValueError(
                    f"cache group {spec.group_id!r} has no planned fields; "
                    "every published group must appear in the memory plan"
                )
            aligned.append(
                replace(
                    spec,
                    cache_blocks_per_lcm_block=group.cache_blocks_per_lcm_block,
                )
            )
            counts[spec.group_id] = group.page_count
        self.paged_cache_group_specs = tuple(aligned)
        self.paged_cache_group_page_counts = counts
        self.runtime_contract = PagedCacheRuntimeContract(
            block_size=self.page_size,
            num_lcm_blocks=self.plan.num_lcm_blocks,
            token_capacity=token_capacity,
            group_specs=self.paged_cache_group_specs,
            group_page_counts=counts,
            external_group_ids=tuple(
                sorted({field.group_id for field in self.plan.fields if field.external})
            ),
        )

    def field(self, field_id: str, dtype: torch.dtype) -> torch.Tensor:
        """Return one typed field view into the shared cache buffer."""
        buffer = self._ensure_buffer()
        view = self._fields.get(field_id)
        if view is not None:
            if view.dtype != dtype:
                raise ValueError(
                    f"cache field {field_id!r} is already bound as {view.dtype}"
                )
            return view
        try:
            field = self.plan.field(field_id)
        except KeyError as exc:
            raise ValueError(f"cache field {field_id!r} is not planned") from exc
        if torch.empty((), dtype=dtype).element_size() != field.element_size:
            raise ValueError(f"field {field_id!r}: dtype itemsize does not match plan")
        group = self.plan.group(field.group_id)
        if field.external:
            if self.plan.max_state_slots > 0:
                # One buffer, two ends: the parents grow up from offset 0 and
                # these rows down from the end, so the boundary between them is
                # a pair of quotas rather than a reallocation. Addresses never
                # move, which a CUDA graph requires -- it bakes `data_ptr()`
                # into its kernel arguments at capture and never re-reads them.
                view = external_field_view(buffer, self.plan, field_id, dtype)
            else:
                # No state region declared: its own dense tensor, one
                # contiguous row per kernel page id. The id comes from the
                # scheduler's second block pool; see ``CacheFieldSpec.external``.
                view = torch.zeros(
                    (group.page_count, *field.shape), dtype=dtype, device=self.device
                )
            self._fields[field_id] = view
            return view
        element_strides = []
        stride = 1
        for extent in reversed(field.shape):
            element_strides.append(stride)
            stride *= extent
        view = buffer.view(dtype).as_strided(
            (group.page_count, *field.shape),
            (
                field.page_stride_bytes // field.element_size,
                *reversed(element_strides),
            ),
            self._field_block_byte_offset(field_id, 0) // field.element_size,
        )
        self._fields[field_id] = view
        return view

    def zero_blocks(self, block_ids_by_group: dict[str, list[int]]) -> None:
        """Clear selected CacheBlocks without interpreting their field types."""
        buffer = self._ensure_buffer()
        segments = [
            segment
            for group_id, block_ids in block_ids_by_group.items()
            for segment in self._block_byte_segments(group_id, block_ids)
        ]
        if segments:
            zero_byte_ranges(buffer, segments)
        for group_id, block_ids in block_ids_by_group.items():
            if not block_ids:
                continue
            rows = None
            for field in self.plan.fields:
                if not field.external or field.group_id != group_id:
                    continue
                view = self._fields.get(field.field_id)
                if view is None:
                    continue
                if rows is None:
                    rows = torch.tensor(
                        block_ids, dtype=torch.int64, device=view.device
                    )
                view.index_fill_(0, rows, 0)

    @property
    def supports_disaggregation(self) -> bool:
        """True when this pool can move its KV over the PD cache-group contract.

        Enabled by --disaggregation-mode; every contract-planned pool exposes
        the same raw-slab transfer ABI (get_pd_cache_contract), so PD is a
        uniform capability rather than a per-family special case.
        """
        return self._pd_disaggregation_enabled

    def get_pd_cache_contract(self):
        """Describe the LCM arena for PD transfer (layout + slab registrations)."""
        if not self.supports_disaggregation:
            raise RuntimeError(
                "paged cache PD requires --disaggregation-mode on this pool"
            )
        return self.pd_contract(self.paged_cache_group_specs)

    def pd_contract(self, group_specs):
        buffer = self._ensure_buffer()
        from tokenspeed.runtime.pd.cache_protocol import build_lcm_pd_cache_contract

        missing = [
            field.field_id
            for field in self.plan.fields
            if field.field_id not in self._fields
        ]
        if missing:
            raise RuntimeError(f"cache fields have no runtime dtype: {missing}")
        field_dtypes = {
            field_id: str(view.dtype).removeprefix("torch.")
            for field_id, view in self._fields.items()
        }
        return build_lcm_pd_cache_contract(
            plan=self.plan,
            buffer=buffer,
            group_specs=group_specs,
            field_dtypes=field_dtypes,
        )

    def _ensure_buffer(self) -> torch.Tensor:
        if self._backing_pool is not None:
            buffer = self._backing_pool.buffer
            if buffer is None:
                raise RuntimeError("the backing cache pool released its buffer")
            self.buffer = buffer
            return buffer
        if self.buffer is None:
            self.buffer = torch.zeros(
                # Equals `arena_bytes` until a plan declares a state region, so
                # a model without one allocates exactly what it always did.
                self.plan.unified_buffer_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
            # The startup memory table and the pool's reported `size` are both
            # derived, not measured, and they disagree with the arena by a
            # factor of two on gemma-4. Print what was actually allocated.
            logger.info(
                "KV arena allocated: %.2f GB (%d bytes)",
                self.buffer.numel() / 1e9,
                self.buffer.numel(),
            )
        return self.buffer

    def _field_block_byte_offset(self, field_id: str, block_id: int) -> int:
        field = self.plan.field(field_id)
        if field.external:
            raise ValueError(
                f"cache field {field_id!r} is external and has no arena offset"
            )
        group = self.plan.group(field.group_id)
        if block_id < 0 or block_id >= group.page_count:
            raise IndexError(
                f"block_id {block_id} outside [0, {group.page_count}) for "
                f"group {group.group_id!r}"
            )
        plane = self.plan.plane(field.plane_id)
        return (
            plane.arena_offset_bytes
            + plane.bytes_per_lcm_block
            - field.page_stride_bytes
            + block_id * field.page_stride_bytes
            + field.field_offset_bytes
        )

    def _block_byte_segments(
        self, group_id: str, block_ids: list[int]
    ) -> list[tuple[int, int]]:
        self.plan.group(group_id)
        # External fields are not in the arena buffer; zero_blocks clears them
        # on their own tensors.
        fields = [
            field
            for field in self.plan.fields
            if field.group_id == group_id and not field.external
        ]
        return [
            (
                self._field_block_byte_offset(field.field_id, block_id),
                field.payload_bytes,
            )
            for block_id in block_ids
            for field in fields
        ]

    def register_layerwise_load_tracker(
        self, layerwise_load_tracker: LayerwiseLoadTracker
    ) -> None:
        self.layerwise_load_tracker = layerwise_load_tracker

    def bind_paged_cache_scheduler(self, scheduler: object) -> None:
        """Keep the scheduler handle for paged-cache diagnostics."""
        self._paged_cache_scheduler = scheduler

    def cache_transfer_layout(self):
        """Return the byte contract used by Host cache transfers."""
        from tokenspeed.runtime.cache.transfer.layout import (
            layout_from_lcm_plan,
            select_layer_fields,
        )

        try:
            layer_num = self.layer_num
        except AttributeError as exc:
            raise RuntimeError(
                f"{type(self).__name__} must expose layer_num for Host L2"
            ) from exc
        try:
            field_ids, consumers = select_layer_fields(
                self.plan.fields,
                first_layer=self._field_layer_offset,
                num_layers=layer_num,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        local_group_ids = {
            field.group_id for field in self.plan.fields if field.field_id in field_ids
        }
        scheduler_group_ids = tuple(
            spec.group_id
            for spec in self.paged_cache_group_specs
            if spec.group_id in local_group_ids
        )
        return layout_from_lcm_plan(
            self.plan,
            self._ensure_buffer(),
            consumers=consumers,
            group_ids=scheduler_group_ids or None,
            field_ids=field_ids,
        )

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        """Zero the shared cache buffer after sleep/wake remaps its storage."""
        # The event loop visits both target and draft pools. A draft view owns
        # no allocation; the target clears their shared arena exactly once.
        if self._backing_pool is not None:
            return
        if self.buffer is not None:
            self.buffer.zero_()

    def maybe_log_paged_cache_group_pages(self) -> None:
        """Report per-group page occupancy when asked for it.

        The aggregate `#pages(active/cached/total)` line cannot answer the
        question that matters for a windowed model: whether the sliding groups
        release what has slid out of the window, or hold the whole span like the
        full-attention group does. Those two cases differ by a large factor in
        how many concurrent requests fit, and look identical in the aggregate.

        Off unless `TOKENSPEED_LOG_GROUP_PAGES=1`, because it prints once per
        decode-batch log line.
        """

        scheduler = getattr(self, "_paged_cache_scheduler", None)
        if (
            getattr(self, "rank", 0) != 0
            or scheduler is None
            or not self.paged_cache_group_specs
            or os.getenv("TOKENSPEED_LOG_GROUP_PAGES") != "1"
        ):
            return
        parts = []
        for spec in self.paged_cache_group_specs:
            group_id = spec.group_id
            try:
                total = scheduler.paged_cache_group_total_pages(group_id)
                available = scheduler.paged_cache_group_available_pages(group_id)
                # `available` is dominated by the arena-wide empty-parent count
                # scaled by this group's packing, so `total - available` is the
                # same fraction for every group and says nothing about which
                # group holds what -- do not subtract anything from it. Read
                # `parents` for occupancy and `cached` for how much of that is
                # retention: on Qwen3.8-27B c=16 the three recurrent groups
                # report parents=68/69 with cached=60/61, so a state group's
                # parents are retained snapshots, not the pages running
                # requests are writing.
                parents = scheduler.paged_cache_group_bound_parents(group_id)
                # Pages held as cache. `used - cached` is the live footprint,
                # and the two need different fixes: a group whose parents are
                # nearly all live is bounded by concurrency, one whose parents
                # are nearly all cached is bounded by retention.
                cached = scheduler.paged_cache_group_cached_pages(group_id)
            except Exception:  # diagnostics must never take the engine down
                return
            window = spec.sliding_window_tokens or 0
            # Cached-but-unreachable. A sliding group publishes its boundary run
            # at exactly BoundaryLookbackBlocks() pages and a hit needs exactly
            # that many contiguous ones, so evicting a single page out of a run
            # strands the rest unless the run reaches block 0. `cached` alone
            # cannot show that: a fragmented group and an intact one report the
            # same number.
            try:
                prof = scheduler.paged_cache_group_run_profile(group_id)
                frag = f", runs={prof[3]}, stranded={prof[4]}/{prof[2]}, depth={prof[5]}"
            except Exception:
                frag = ""
            parts.append(
                f"{group_id}[{spec.retention}"
                + (f",w={window}" if window else "")
                + f"]: parents={parents}, cached={cached}, used={total - available}/{total}{frag}"
            )
        # Which arena admission could not satisfy. The split between them is a
        # function of the workload -- measured, agent c=16 wants 112 state
        # pages per group and collapses to 0.55x at 180, while RAG c=128 wants
        # 180 and sits at 0.69x with 112 -- so this is what a rebalance reads.
        try:
            shortages = scheduler.arena_shortages()
        except Exception:
            shortages = None
        suffix = f" | arena refusals {list(shortages)}" if shortages else ""
        logger.info("Paged-cache group pages. %s%s", "; ".join(parts), suffix)
        self._maybe_warn_arena_split(shortages)

    # Refusals have to be this lopsided before the split is called wrong. Below
    # it the two arenas are both occasionally short, which is what a balanced
    # split looks like.
    _ARENA_SPLIT_RATIO = 8.0

    def _maybe_warn_arena_split(self, shortages) -> None:
        """Say which arena is starving, once, when the split is clearly wrong.

        The right `--state-arena-pages` is a function of the workload and cannot
        be a constant. Measured on Qwen3.8-27B, device-only: the agent shape at
        c=16 runs 1.05x of vLLM with 112 pages per group and 0.55x with 180,
        while RAG at c=128 runs 0.69x with 112 and 1.41x with 180. The engine
        cannot know which it is serving before requests arrive, but it can see
        which arena admission could not satisfy, and that is enough to name the
        direction.
        """
        if (
            not shortages
            or len(shortages) < 2
            or getattr(self, "_arena_split_warned", False)
        ):
            return
        attention, state = shortages[0], max(shortages[1:])
        total = attention + state
        if total < 1000:  # too early to read
            return
        if attention > state * self._ARENA_SPLIT_RATIO:
            direction, starving = "lower", "attention"
        elif state > attention * self._ARENA_SPLIT_RATIO:
            direction, starving = "raise", "recurrent-state"
        else:
            return
        self._arena_split_warned = True
        logger.warning(
            "Cache arena split looks wrong for this workload: the %s arena was "
            "short on %d admissions against %d for the other. %s "
            "--state-arena-pages. A state page is far larger than an attention "
            "page, so the two arenas trade directly: the right split depends on "
            "the prompt length and the concurrency being served, not on the "
            "model.",
            starving,
            attention if starving == "attention" else state,
            state if starving == "attention" else attention,
            direction.capitalize(),
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    # Buffer metadata used by prefill/decode disaggregation.
    def get_contiguous_buf_infos(self):
        raise NotImplementedError()

    def get_contiguous_buf_unit_lens(self):
        return [1] * len(self.get_contiguous_buf_infos()[2])

    # Layerwise buffer offsets used by prefill/decode disaggregation.
    def get_layerwise_buf_info_offsets(self, start_idx=0):
        raise NotImplementedError()


class LayerMappedKVPool:
    """Wraps a KV pool to map the caller's layer IDs to inner pool indices.

    Two callers, one mechanism — a dict from the id a model layer carries to
    the index its plane occupies in the wrapped pool:

    - Hybrid models: layers carry global sparse ids (e.g. 3, 7, 11) while the
      inner pool holds compact full-attention planes (0, 1, 2); the map is
      ``{global_id: pool_idx}`` (the default built from ``layer_ids``).
    - Draft views of the ONE merged pool: draft layers carry LOCAL ids
      (0..n-1) while their planes are the continuation range
      ``num_target_layers..``; the map is ``{local: global}`` (pass
      ``layer_map`` explicitly).
    """

    def __init__(
        self,
        inner_pool,
        full_attention_layer_ids: list[int],
        *,
        layer_map: dict[int, int] | None = None,
    ):
        self.inner = inner_pool
        self.layer_ids = list(full_attention_layer_ids)
        self.layer_map = (
            dict(layer_map)
            if layer_map is not None
            else {
                global_id: pool_idx
                for pool_idx, global_id in enumerate(full_attention_layer_ids)
            }
        )
        # Expose page_size from inner pool for the scheduler
        self.page_size = getattr(inner_pool, "page_size", 1)

    def _map(self, layer_id: int) -> int:
        return self.layer_map.get(layer_id, layer_id)

    @contextmanager
    def _mapped(self, layer):
        """Temporarily remap ``layer.layer_id`` to its inner-pool slot."""
        orig = layer.layer_id
        layer.layer_id = self._map(orig)
        try:
            yield
        finally:
            layer.layer_id = orig

    def set_kv_buffer(
        self,
        layer,
        out_cache_loc: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor | None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ):
        with self._mapped(layer):
            self.inner.set_kv_buffer(layer, out_cache_loc, k, v, k_scale, v_scale)

    def get_kv_buffer(self, layer_id: int):
        return self.inner.get_kv_buffer(self._map(layer_id))

    def get_key_buffer(self, layer_id: int):
        return self.inner.get_key_buffer(self._map(layer_id))

    def get_value_buffer(self, layer_id: int):
        return self.inner.get_value_buffer(self._map(layer_id))

    # MLA pools index their per-layer kv_buffer by ``layer.layer_id`` directly.
    # In a hybrid model the inner MLA pool only holds the full-attention layers,
    # so the global id must be mapped to its pool slot first (mirrors
    # ``set_kv_buffer``). Reached via the DeepseekV3-style MLA chunked-prefill
    # path (Kimi-K3).
    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope, sanitize=True):
        # Prefill breakable-graph padding contract: the dummy-batch capture (and
        # bucket-padding rows) whose ``out_cache_loc`` is the reserved
        # ``dummy_kv_slot`` can carry NaN into this fp8 KV write. The paged MLA
        # decode kernel reads that shared dummy slot through the zero-padded
        # block-table entries and computes ``q·k`` BEFORE applying the causal
        # mask, so the NaN survives it (``NaN + -inf = NaN``) and poisons a live
        # row's softmax -> NaN logits -> token 0. Eager prefill leaves the dummy
        # slot finite (``q·0`` masks cleanly), which is why the bug only appears
        # with the prefill graph on. Ask the cache writer to sanitize in-kernel
        # so real rows stay bitwise unchanged without allocating two temporary
        # tensors or launching two separate nan_to_num kernels.
        with self._mapped(layer):
            self.inner.set_mla_kv_buffer(
                layer,
                loc,
                cache_k_nope,
                cache_k_rope,
                sanitize=sanitize,
            )

    def get_mla_kv_buffer(self, layer, loc, dst_dtype=None):
        with self._mapped(layer):
            return self.inner.get_mla_kv_buffer(layer, loc, dst_dtype)

    # DSA/MSA index-key planes are layer-indexed like the KV planes; a draft
    # view must map its local layer ids onto the continuation range here too
    # (an unmapped pass-through would read/write the target's planes).
    def get_index_k_buffer(self, layer_id: int):
        return self.inner.get_index_k_buffer(self._map(layer_id))

    def set_index_k_buffer(self, layer_id: int, loc, index_k):
        self.inner.set_index_k_buffer(self._map(layer_id), loc, index_k)

    def gather_index_k(self, layer_id: int, slots):
        return self.inner.gather_index_k(self._map(layer_id), slots)

    def kvconv_checkpoint_buffers(self, layer_id: int):
        return self.inner.kvconv_checkpoint_buffers(self._map(layer_id))

    def hiddenconv_checkpoint_buffer(self, layer_id: int, component: str):
        return self.inner.hiddenconv_checkpoint_buffer(self._map(layer_id), component)

    def __getattr__(self, name):
        return getattr(self.inner, name)
