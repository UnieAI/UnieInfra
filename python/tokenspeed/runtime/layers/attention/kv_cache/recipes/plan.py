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

"""Pure integer geometry for one shared LCM cache arena."""

from __future__ import annotations

import logging
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

# Planner/runtime limits, not model layout inputs.
_MAX_LCM_BLOCK_BYTES = (1 << 63) - 1
_MAX_KERNEL_PAGE_ID = (1 << 31) - 1


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheGroupLayout:
    group_id: str
    cache_blocks_per_lcm_block: int
    page_count: int


@dataclass(frozen=True)
class CachePlaneLayout:
    plane_id: str
    bytes_per_lcm_block: int
    arena_offset_bytes: int


@dataclass(frozen=True)
class CacheFieldLayout:
    group_id: str
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    element_size: int
    field_offset_bytes: int
    page_stride_bytes: int
    # Bytes live in a standalone tensor. See ``CacheFieldSpec.external``.
    external: bool = False

    @property
    def payload_bytes(self) -> int:
        return math.prod(self.shape) * self.element_size


@dataclass(frozen=True)
class CacheMemoryPlan:
    """Static byte geometry for one shared physical LCM arena.

    ``num_lcm_blocks`` excludes the null parent. ``arena_bytes`` includes it.
    """

    logical_block_tokens: int
    lcm_block_bytes: int
    num_lcm_blocks: int
    groups: tuple[CacheGroupLayout, ...]
    planes: tuple[CachePlaneLayout, ...] = ()
    fields: tuple[CacheFieldLayout, ...] = ()

    @property
    def arena_bytes(self) -> int:
        return (self.num_lcm_blocks + 1) * self.lcm_block_bytes

    def group(self, group_id: str) -> CacheGroupLayout:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(group_id)

    def field(self, field_id: str) -> CacheFieldLayout:
        for field in self.fields:
            if field.field_id == field_id:
                return field
        raise KeyError(field_id)

    def plane(self, plane_id: str) -> CachePlaneLayout:
        for plane in self.planes:
            if plane.plane_id == plane_id:
                return plane
        raise KeyError(plane_id)

    def capacity_report(
        self,
        *,
        window_tokens: Mapping[str, int] | None = None,
        per_request_blocks: Mapping[str, int] | None = None,
        max_num_seqs: int | None = None,
    ) -> dict:
        """Per-group capacity in its own consumption unit, plus dead bytes.

        Heterogeneous groups consume in different units — full-attention
        per token, state (KDA/Mamba) per request, sliding-window bounded by
        the window — so a single token-denominated capacity misstates two
        of the three. Callers name the special groups:

        Args:
            window_tokens: retention-bounded groups (sliding window): active
                demand per request is at most the window, so capacity beyond
                ``max_num_seqs × window`` is dead rows (stranded by the
                static slab split).
            per_request_blocks: per-request-constant groups (state): the
                admission they support is ``page_count / blocks_per_req``.
            max_num_seqs: admission bound used for dead-row estimates.

        Returns:
            ``{group_id: {"unit", "capacity", "supported_requests",
            "dead_bytes", "binding_utilization"}}`` with
            ``supported_requests`` None when unknown. Binding admission =
            min over non-None supported_requests. ``binding_utilization``
            is the fraction of a parent a binding of this group actually
            uses (aliased slabs are sized by their widest tenant; a
            narrower group's binding leaves the rest dead — the
            binding hole).
        """
        window_tokens = dict(window_tokens or {})
        per_request_blocks = dict(per_request_blocks or {})
        group_bytes_per_block: dict[str, int] = {}
        for field in self.fields:
            group_bytes_per_block[field.group_id] = (
                group_bytes_per_block.get(field.group_id, 0) + field.payload_bytes
            )
        report: dict[str, dict] = {}
        for group in self.groups:
            usable_pages = self.num_lcm_blocks * group.cache_blocks_per_lcm_block
            block_bytes = group_bytes_per_block.get(group.group_id, 0)
            binding_utilization = (
                group.cache_blocks_per_lcm_block * block_bytes / self.lcm_block_bytes
                if self.lcm_block_bytes
                else 0.0
            )
            if group.group_id in per_request_blocks:
                blocks_per_req = max(1, per_request_blocks[group.group_id])
                supported = usable_pages // blocks_per_req
                demand_pages = max_num_seqs * blocks_per_req if max_num_seqs else None
                report[group.group_id] = {
                    "unit": "requests",
                    "capacity": supported,
                    "supported_requests": supported,
                    "dead_bytes": (
                        max(0, usable_pages - demand_pages) * block_bytes
                        if demand_pages is not None
                        else None
                    ),
                    "binding_utilization": binding_utilization,
                }
                continue
            token_capacity = usable_pages * self.logical_block_tokens
            if group.group_id in window_tokens:
                window = max(1, window_tokens[group.group_id])
                demand_tokens = max_num_seqs * window if max_num_seqs else None
                supported = token_capacity // window
                report[group.group_id] = {
                    "unit": "tokens",
                    "capacity": token_capacity,
                    "supported_requests": supported,
                    "dead_bytes": (
                        max(0, token_capacity - demand_tokens)
                        // self.logical_block_tokens
                        * block_bytes
                        if demand_tokens is not None
                        else None
                    ),
                    "binding_utilization": binding_utilization,
                }
                continue
            report[group.group_id] = {
                "unit": "tokens",
                "capacity": token_capacity,
                # Depends on per-request context length; unknown here.
                "supported_requests": None,
                "dead_bytes": 0,
                "binding_utilization": binding_utilization,
            }
        return report


@dataclass(frozen=True)
class CacheFieldSpec:
    group_id: str
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    element_size: int
    # True when the field's kernel walks pages by an implicit payload-sized
    # stride. False when the kernel consumes the tensor's runtime stride.
    exact_page_stride: bool = True
    # Some kernels accept padded pages but still require the runtime page
    # stride to satisfy an alignment constraint (for example, a TMA row
    # stride). The planner applies this in bytes after group packing.
    page_stride_alignment_bytes: int = 1
    # This field's bytes live in a standalone tensor, not in an arena plane.
    #
    # A cache block resolves to a dense kernel page id --
    # `1 + (lcm_block_id - 1) * cache_blocks_per_lcm_block + slot_index` -- so a
    # tensor of `pages + 1` rows is addressed by exactly the id the allocator
    # produces. The page id and the byte location are separable, and separating
    # them is what lets a group sit in its own arena.
    #
    # The LCM parent is sized by the widest geometry family, so a group whose
    # page is narrower strands the difference on every parent it binds. On
    # Qwen3.8-27B a GDN state page is 48.94 MiB against a 96 MiB parent and
    # packs 1, so the three recurrent groups take 206 of 346 parents -- 60 of
    # every 68 of them retained snapshots -- and full attention is left 188,928
    # tokens of prefix cache against vLLM's 447,146. No packing fixes it: the
    # parent is the sum of the attention family's planes, which are twice the
    # recurrent family's (k and v per layer against one state), so the ratio is
    # 16/32 for every packing and every parent size.
    #
    # An external group must be given both its packing and its page count: it
    # has no plane footprint for the byte-ratio solve to read, and its pages
    # come from the scheduler's second BlockPool, not from `num_lcm_blocks`.
    external: bool = False

    @property
    def payload_bytes(self) -> int:
        return math.prod(self.shape) * self.element_size


@dataclass(frozen=True)
class CacheLayout:
    """Capacity-independent byte geometry for one LCM block."""

    logical_block_tokens: int
    lcm_block_bytes: int
    group_packing: tuple[tuple[str, int], ...]
    plane_bytes: tuple[tuple[str, int], ...]
    fields: tuple[CacheFieldLayout, ...]

    def with_num_lcm_blocks(
        self, num_lcm_blocks: int, external_pages: Mapping[str, int] | None = None
    ) -> CacheMemoryPlan:
        if (
            isinstance(num_lcm_blocks, bool)
            or not isinstance(num_lcm_blocks, int)
            or num_lcm_blocks < 1
        ):
            raise ValueError("num_lcm_blocks must be a positive integer")

        external_pages = dict(external_pages or {})
        external_groups = {field.group_id for field in self.fields if field.external}
        unknown = set(external_pages) - external_groups
        if unknown:
            raise ValueError(
                f"external_pages names groups with no external field: {sorted(unknown)}"
            )
        missing = sorted(external_groups - set(external_pages))
        if missing:
            raise ValueError(
                "an external group draws from the scheduler's second block pool "
                f"and must be given its page count: {missing}"
            )
        groups = []
        for group_id, count in self.group_packing:
            # An external group's pages come from its own arena, so its page-id
            # space is that arena's parents times its packing -- not this one's.
            blocks = external_pages.get(group_id, num_lcm_blocks)
            if blocks * count > _MAX_KERNEL_PAGE_ID:
                raise ValueError(
                    f"cache group {group_id!r}: kernel page id exceeds "
                    f"{_MAX_KERNEL_PAGE_ID}"
                )
            groups.append(
                CacheGroupLayout(
                    group_id=group_id,
                    cache_blocks_per_lcm_block=count,
                    page_count=1 + blocks * count,
                )
            )

        # Planes of DIFFERENT geometry families overlay the same arena bytes.
        #
        # A parent is bound to exactly one group for as long as it holds any
        # block (`block_pool.h` asserts "LCM parent binding changed while
        # occupied", and `bound_group.reset()` runs only at occupied_count == 0,
        # so no cached block outlives the binding either). A group only ever
        # touches its own geometry's planes. So the planes of two families are
        # never both live for the same parent id, and giving each family its own
        # arena region only reserves bytes that parent can never use.
        #
        # Laying every plane out consecutively made the parent the UNION of all
        # families. Measured on gemma-4-31B: 40 planes (10 units x {4x512.k,
        # 4x512.v, k, v}), lcm_block_bytes 20,971,520, and the engine's own
        # `binding_utilization` reporting 0.5 for all six groups -- 13.2 GiB of a
        # 26.4 GiB arena dead from the moment it was allocated. The cost is
        # 1 - 1/families, so it appears only on models whose layers disagree
        # about KV geometry, which is why a single-geometry model never saw it.
        #
        # Overlaying makes the parent the MAX over families instead. Same budget,
        # 2x the parents on a two-family model.
        plane_family: dict[str, int] = {}
        family_of_group: dict[str, int] = {}
        for field in self.fields:
            if field.external:
                continue
            fam = family_of_group.get(field.group_id)
            if fam is None:
                fam = plane_family.get(field.plane_id)
            if fam is None:
                fam = len(set(plane_family.values()))
            family_of_group[field.group_id] = fam
            plane_family[field.plane_id] = fam
        # Overlaying is only sound when every plane has the SAME size. The
        # layout is plane-major: parent k sits at `plane_offset + k *
        # plane_bytes`. Two families that start at the same arena offset with
        # DIFFERENT plane sizes put their parent k at different byte ranges, so
        # parent k of one family lands inside parent 2k of the other -- and
        # `BoundGroup` cannot catch it, because each parent really is bound to
        # exactly one group. It is a silent wrong read, not a violated
        # invariant.
        #
        # gemma-4 satisfies it (40 planes, all 524,288) which is why the
        # overlay is correct there. Measured what happens when it does not:
        # splitting Qwen3.5-27B's plane namespaces by layer kind gives FULL
        # planes of 1,572,864 against LIN planes of 3,145,728, and the server
        # answers correctly for eleven requests and then returns "!!!!!!!!"
        # for every request after that, with no error anywhere.
        #
        # Fall back to consecutive (union) layout rather than refuse to serve:
        # a union parent wastes bytes but reads the right ones.
        distinct_plane_sizes = {size for _, size in self.plane_bytes}
        overlay = len(set(plane_family.values())) <= 1 or len(distinct_plane_sizes) == 1
        if not overlay and len(set(plane_family.values())) > 1:
            logger.warning(
                "Cache arena: %d geometry families but %d distinct plane "
                "sizes %s -- laying families out consecutively instead of "
                "overlaying them. The parent is the UNION of the families and "
                "every parent wastes 1 - 1/families of its bytes. Overlaying "
                "needs one plane size across all families; see the comment "
                "above this check.",
                len(set(plane_family.values())),
                len(distinct_plane_sizes),
                sorted(distinct_plane_sizes),
            )
        planes = []
        family_offset: dict[int, int] = {}
        for plane_id, bytes_per_lcm_block in self.plane_bytes:
            fam = plane_family.get(plane_id, 0) if overlay else 0
            arena_offset = family_offset.get(fam, 0)
            planes.append(
                CachePlaneLayout(
                    plane_id=plane_id,
                    bytes_per_lcm_block=bytes_per_lcm_block,
                    arena_offset_bytes=arena_offset,
                )
            )
            family_offset[fam] = (
                arena_offset + (num_lcm_blocks + 1) * bytes_per_lcm_block
            )

        return CacheMemoryPlan(
            logical_block_tokens=self.logical_block_tokens,
            lcm_block_bytes=self.lcm_block_bytes,
            num_lcm_blocks=num_lcm_blocks,
            groups=tuple(groups),
            planes=tuple(planes),
            fields=self.fields,
        )


def merge_continuation_layers(
    *,
    fields,
    layer_types: tuple[str, ...],
    group_ids: tuple[str, ...],
    layer_kv_head_counts: tuple[int, ...] | None = None,
    draft_fields=None,
    draft_layer_types: tuple[str, ...] = (),
    draft_group_ids: tuple[str, ...] = (),
    draft_layer_kv_head_counts: tuple[int, ...] | None = None,
) -> tuple:
    """Merge a draft model's per-layer vectors after the target's.

    One big model: the draft's fields renumber via
    :func:`continue_layer_fields`; every other per-layer vector is plain
    concatenation. Returns ``(fields, layer_types, group_ids,
    layer_kv_head_counts, num_draft_layers)`` — all target-shaped, so
    downstream builders stay draft-oblivious.
    """
    merged_fields = tuple(fields)
    merged_layer_types = tuple(layer_types)
    merged_group_ids = tuple(group_ids)
    merged_head_counts = layer_kv_head_counts
    num_draft_layers = 0
    if draft_fields is not None:
        num_target_layers = len(merged_group_ids)
        num_draft_layers = len(draft_group_ids)
        if len(draft_layer_types) != num_draft_layers:
            raise ValueError(
                f"draft layer_types has {len(draft_layer_types)} entries but "
                f"draft group_ids has {num_draft_layers}"
            )
        merged_fields += continue_layer_fields(
            draft_fields, first_layer_id=num_target_layers
        )
        merged_layer_types += tuple(draft_layer_types)
        merged_group_ids += tuple(draft_group_ids)
        if bool(layer_kv_head_counts) != bool(draft_layer_kv_head_counts):
            raise ValueError(
                "layer_kv_head_counts must be supplied for both sides or neither"
            )
        if layer_kv_head_counts is not None:
            merged_head_counts = tuple(layer_kv_head_counts) + tuple(
                draft_layer_kv_head_counts
            )
    return (
        merged_fields,
        merged_layer_types,
        merged_group_ids,
        merged_head_counts,
        num_draft_layers,
    )


def continue_layer_fields(fields, *, first_layer_id: int) -> tuple[CacheFieldSpec, ...]:
    """Renumber per-layer fields as continuation layers of one big model.

    Draft layers join the target's ``solve_cache_layout`` as ordinary
    layers of the ONE merged model:
    a draft model's local ``layer.{i}...`` field/plane ids become the
    global ``layer.{first_layer_id + i}...``. Group ids are untouched: a
    draft layer in a target group shares its page-id space and packing by
    construction. No draft-specific namespace exists — the merged plan is
    simply a model with more layers.
    """
    renumber = re.compile(r"^(?P<head>layer|unit|slot)\.(?P<idx>\d+)")

    def _shift(identifier: str) -> str:
        return renumber.sub(
            lambda m: f"{m.group('head')}.{int(m.group('idx')) + first_layer_id}",
            identifier,
        )

    return tuple(
        CacheFieldSpec(
            field.group_id,
            _shift(field.field_id),
            _shift(field.plane_id),
            field.shape,
            field.element_size,
            exact_page_stride=field.exact_page_stride,
            page_stride_alignment_bytes=field.page_stride_alignment_bytes,
        )
        for field in fields
    )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _solve_packing(fields):
    """Derive per-group cache blocks per LCM block from exact field ratios."""
    exact_by_plane: dict[str, dict[str, int]] = {}
    groups = set()
    for field in fields:
        groups.add(field.group_id)
        if field.exact_page_stride:
            plane = exact_by_plane.setdefault(field.plane_id, {})
            plane[field.group_id] = plane.get(field.group_id, 0) + field.payload_bytes

    # ratio[g] = (num, den) expressing K_g as a multiple of its component root.
    root = {group_id: group_id for group_id in groups}
    ratio = {group_id: (1, 1) for group_id in groups}

    def resolve(group_id):
        num, den = 1, 1
        while root[group_id] != group_id:
            r_num, r_den = ratio[group_id]
            num, den = num * r_num, den * r_den
            group_id = root[group_id]
        return group_id, (num, den)

    for plane_id in sorted(exact_by_plane):
        tenants = sorted(exact_by_plane[plane_id].items())
        for (first_id, first_bytes), (other_id, other_bytes) in pairwise(tenants):
            first_root, (f_num, f_den) = resolve(first_id)
            other_root, (o_num, o_den) = resolve(other_id)
            if first_root == other_root:
                if f_num * o_den * first_bytes != o_num * f_den * other_bytes:
                    return None
                continue
            num = f_num * o_den * first_bytes
            den = o_num * f_den * other_bytes
            common = math.gcd(num, den)
            root[other_root] = first_root
            ratio[other_root] = (num // common, den // common)

    members: dict[str, list] = {}
    for group_id in sorted(groups):
        component, fraction = resolve(group_id)
        members.setdefault(component, []).append((group_id, fraction))

    packing: dict[str, int] = {}
    for component_members in members.values():
        if len(component_members) == 1:
            continue
        scale = 1
        for _, (_, den) in component_members:
            scale = scale * den // math.gcd(scale, den)
        counts = {
            group_id: num * scale // den for group_id, (num, den) in component_members
        }
        smallest = 0
        for count in counts.values():
            smallest = math.gcd(smallest, count)
        packing.update(
            {group_id: count // smallest for group_id, count in counts.items()}
        )

    return packing


def _packing_by_group_ratio(raw_by_group):
    largest_payload = max(raw_by_group.values())
    return {
        group_id: max(1, largest_payload // raw_by_group[group_id])
        for group_id in raw_by_group
    }


def _check_exact_page_strides(fields, plane_bytes, packing):
    for field in fields:
        if not field.exact_page_stride:
            continue
        stride = plane_bytes[field.plane_id] // packing[field.group_id]
        if stride != field.payload_bytes:
            widest = max(
                (
                    (other.payload_bytes * packing[other.group_id], other.field_id)
                    for other in fields
                    if other.plane_id == field.plane_id
                ),
                default=(0, ""),
            )
            raise ValueError(
                f"cache field {field.field_id!r} needs page stride "
                f"{field.payload_bytes} but plane {field.plane_id!r} gives "
                f"{stride}; its kernel indexes pages by an implicit "
                f"payload-sized stride and would read the wrong rows. The "
                f"plane is sized by {widest[1]!r}; move one of them to another "
                "plane or align their payloads"
            )


def solve_cache_layout(
    fields,
    *,
    logical_block_tokens,
    cache_blocks_per_lcm_block: Mapping[str, int] | None = None,
    alignment=1,
    max_padding_fraction=0.25,
):
    """Solve one capacity-independent, plane-major LCM block layout."""
    if logical_block_tokens <= 0:
        raise ValueError("logical_block_tokens must be > 0")
    if alignment <= 0:
        raise ValueError("alignment must be > 0")
    if max_padding_fraction < 0:
        raise ValueError("max_padding_fraction must be >= 0")

    ordered_fields = tuple(
        sorted(
            fields,
            key=lambda field: (field.plane_id, field.group_id, field.field_id),
        )
    )
    if not ordered_fields:
        raise ValueError("at least one cache field is required")
    if len({field.field_id for field in ordered_fields}) != len(ordered_fields):
        raise ValueError("cache field ids must be unique")

    arena_fields = tuple(field for field in ordered_fields if not field.external)
    external_only_groups = {f.group_id for f in ordered_fields if f.external} - {
        f.group_id for f in arena_fields
    }

    raw_by_group: dict[str, int] = {}
    for field in ordered_fields:
        if not field.group_id or not field.field_id or not field.plane_id:
            raise ValueError(
                "cache field group_id, field_id and plane_id must be non-empty"
            )
        if (
            field.element_size <= 0
            or not field.shape
            or any(extent <= 0 for extent in field.shape)
            or isinstance(field.page_stride_alignment_bytes, bool)
            or not isinstance(field.page_stride_alignment_bytes, int)
            or field.page_stride_alignment_bytes <= 0
        ):
            raise ValueError(f"cache field {field.field_id!r} has invalid geometry")
        if field.external:
            continue
        raw_by_group[field.group_id] = (
            raw_by_group.get(field.group_id, 0) + field.payload_bytes
        )

    if not raw_by_group:
        raise ValueError("cache layout needs at least one arena-resident field")
    ordered_group_ids = tuple(sorted(set(raw_by_group) | external_only_groups))
    # One line, once per plan: the per-group block footprint that decides how
    # many cache blocks share an LCM parent. A group with the largest footprint
    # packs 1:1 and then spends a whole parent per 128-token block, which is
    # what bounds admission on long prompts -- and it is invisible from the
    # token capacity the profile reports.
    if os.environ.get("TOKENSPEED_LOG_CACHE_PACKING"):
        logger.info(
            "cache packing inputs: %s",
            {g: raw_by_group.get(g, 0) for g in ordered_group_ids},
        )
    has_explicit_packing = cache_blocks_per_lcm_block is not None
    if has_explicit_packing:
        unknown = set(cache_blocks_per_lcm_block) - set(ordered_group_ids)
        if unknown:
            raise ValueError(
                "cache_blocks_per_lcm_block names groups outside the plan: "
                f"{sorted(unknown)}"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in cache_blocks_per_lcm_block.values()
        ):
            raise ValueError("cache group packing must be a positive integer")
        # Pinned groups take the caller's count; groups the caller leaves
        # unpinned pack by their byte ratio as in the unpinned solve.
        packing = _packing_by_group_ratio(raw_by_group)
        constrained = _solve_packing(arena_fields)
        if constrained is not None:
            packing.update(constrained)
        packing.update(cache_blocks_per_lcm_block)
    else:
        packing = _packing_by_group_ratio(raw_by_group)
        constrained = _solve_packing(arena_fields)
        if constrained is not None:
            packing.update(constrained)
    missing = sorted(external_only_groups - set(packing))
    if missing:
        raise ValueError(
            "groups whose fields are all external must pin "
            f"cache_blocks_per_lcm_block: {missing}"
        )

    plane_fields: dict[str, dict[str, list[CacheFieldSpec]]] = {}
    for field in arena_fields:
        plane_fields.setdefault(field.plane_id, {}).setdefault(
            field.group_id, []
        ).append(field)

    field_offsets: dict[str, int] = {}
    group_plane_bytes: dict[str, dict[str, int]] = defaultdict(dict)
    for plane_id, by_group in plane_fields.items():
        for group_id, group_fields in by_group.items():
            offset = 0
            for field in group_fields:
                offset = _align_up(offset, field.element_size)
                field_offsets[field.field_id] = offset
                offset += field.payload_bytes
            group_plane_bytes[group_id][plane_id] = offset

    # Flexible fields consume explicit strides and can use slack left by exact
    # fields, but their K must divide every plane they occupy.
    exact_groups = {field.group_id for field in arena_fields if field.exact_page_stride}
    flexible_groups = set(ordered_group_ids) - exact_groups
    if flexible_groups and not has_explicit_packing:
        fixed_plane_bytes = {}
        for plane_id, by_group in plane_fields.items():
            fixed_alignment = alignment
            fixed_required = 0
            for group_id, group_fields in by_group.items():
                if group_id in flexible_groups:
                    continue
                fixed_required = max(
                    fixed_required,
                    packing[group_id] * group_plane_bytes[group_id][plane_id],
                )
                for field in group_fields:
                    required = packing[group_id] * field.element_size
                    fixed_alignment = (
                        fixed_alignment
                        // math.gcd(fixed_alignment, required)
                        * required
                    )
            if fixed_required:
                fixed_plane_bytes[plane_id] = _align_up(
                    fixed_required,
                    fixed_alignment,
                )

        for group_id in sorted(flexible_groups):
            occupied_planes = group_plane_bytes[group_id]
            if not all(plane_id in fixed_plane_bytes for plane_id in occupied_planes):
                continue
            upper = min(
                fixed_plane_bytes[plane_id] // payload_bytes
                for plane_id, payload_bytes in occupied_planes.items()
            )
            element_alignment_by_plane = {}
            for plane_id, group_fields in plane_fields.items():
                fields_for_group = group_fields.get(group_id, ())
                element_alignment = 1
                for field in fields_for_group:
                    element_alignment = (
                        element_alignment
                        // math.gcd(element_alignment, field.element_size)
                        * field.element_size
                    )
                if fields_for_group:
                    element_alignment_by_plane[plane_id] = element_alignment
            for count in range(upper, 0, -1):
                if all(
                    fixed_plane_bytes[plane_id]
                    % (count * element_alignment_by_plane[plane_id])
                    == 0
                    for plane_id in occupied_planes
                ):
                    packing[group_id] = count
                    break

    plane_bytes: dict[str, int] = {}
    for plane_id, by_group in plane_fields.items():
        plane_alignment = alignment
        required_bytes = 0
        for group_id, group_fields in by_group.items():
            required_bytes = max(
                required_bytes,
                packing[group_id] * group_plane_bytes[group_id][plane_id],
            )
            for field in group_fields:
                required = packing[group_id] * field.element_size
                plane_alignment = (
                    plane_alignment // math.gcd(plane_alignment, required) * required
                )
                stride_required = packing[group_id] * field.page_stride_alignment_bytes
                plane_alignment = (
                    plane_alignment
                    // math.gcd(plane_alignment, stride_required)
                    * stride_required
                )
                if plane_alignment > _MAX_LCM_BLOCK_BYTES:
                    raise ValueError(
                        f"LCM block size alignment {plane_alignment} exceeds "
                        f"limit {_MAX_LCM_BLOCK_BYTES}"
                    )
        plane_bytes[plane_id] = _align_up(required_bytes, plane_alignment)

    _check_exact_page_strides(arena_fields, plane_bytes, packing)

    parent_alignment = alignment
    for count in packing.values():
        parent_alignment = parent_alignment // math.gcd(parent_alignment, count) * count
    # MAX over geometry families, not SUM over all planes. A parent is bound to
    # one group while it holds anything, so it only ever touches one family's
    # planes; `CacheMemoryPlan.layout` overlays the families on the same bytes.
    # Summing made the parent the union and left 1 - 1/families of the arena
    # dead -- 13.2 GiB of 26.4 on gemma-4-31B, which the engine was already
    # reporting as binding_utilization 0.5 on every group.
    _fam_of_group: dict[str, int] = {}
    _fam_of_plane: dict[str, int] = {}
    for _f in arena_fields:
        _fam = _fam_of_group.get(_f.group_id, _fam_of_plane.get(_f.plane_id))
        if _fam is None:
            _fam = len(set(_fam_of_plane.values()))
        _fam_of_group[_f.group_id] = _fam
        _fam_of_plane[_f.plane_id] = _fam
    _fam_bytes: dict[int, int] = {}
    for _pid, _b in plane_bytes.items():
        _fam = _fam_of_plane.get(_pid, 0)
        _fam_bytes[_fam] = _fam_bytes.get(_fam, 0) + _b
    lcm_block_bytes = _align_up(max(_fam_bytes.values()), parent_alignment)
    if lcm_block_bytes > _MAX_LCM_BLOCK_BYTES:
        raise ValueError(
            f"LCM block size {lcm_block_bytes} exceeds limit {_MAX_LCM_BLOCK_BYTES}"
        )

    for group_id in ordered_group_ids:
        if group_id in external_only_groups:
            continue
        count = packing[group_id]
        if lcm_block_bytes % count:
            raise ValueError(
                f"cache group {group_id!r}: packing {count} does not partition "
                f"LCM block size {lcm_block_bytes}"
            )
        stride = lcm_block_bytes // count
        padding_fraction = (stride - raw_by_group[group_id]) / raw_by_group[group_id]
        if padding_fraction > max_padding_fraction:
            raise ValueError(
                f"cache group {group_id!r}: padding fraction "
                f"{padding_fraction:.6f} exceeds limit {max_padding_fraction:.6f}"
            )

    field_layouts = []
    for field in ordered_fields:
        field_layouts.append(
            CacheFieldLayout(
                group_id=field.group_id,
                field_id=field.field_id,
                plane_id=field.plane_id,
                shape=field.shape,
                element_size=field.element_size,
                # A standalone tensor starts at its own row 0 and its pages are
                # contiguous, so the offset is 0 and the stride is the payload.
                field_offset_bytes=(
                    0 if field.external else field_offsets[field.field_id]
                ),
                page_stride_bytes=(
                    field.payload_bytes
                    if field.external
                    else plane_bytes[field.plane_id] // packing[field.group_id]
                ),
                external=field.external,
            )
        )

    return CacheLayout(
        logical_block_tokens=logical_block_tokens,
        lcm_block_bytes=lcm_block_bytes,
        group_packing=tuple(
            (group_id, packing[group_id]) for group_id in ordered_group_ids
        ),
        plane_bytes=tuple(
            (plane_id, plane_bytes[plane_id]) for plane_id in sorted(plane_bytes)
        ),
        fields=tuple(field_layouts),
    )
