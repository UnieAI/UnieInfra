# Copyright (c) 2026 UnieAI
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

"""FlashAttention-2 MHA kernels for sm_120, via vLLM's ``vllm_flash_attn`` fork.

Why this exists
---------------
Every FlashAttention family TokenSpeed already knows about is dead on consumer
Blackwell:

* **FA4** (``ops/attention/flash_attn``) is registered with
  ``min_arch == max_arch == (10, 0)`` and needs ``tcgen05`` tensor memory,
  which sm_120 does not have — ``ptxas`` rejects ``tcgen05.st`` outright.
* **FA3** (``flash_attn_interface``, the ``flash_attn_3`` wheel in the image)
  ships 62 ``sm_90`` cubins and **zero PTX**, so it raises
  ``no kernel image is available for execution on the device``. The source path
  in that error says ``hopper/`` and means it.
* **FA2** is not installed at all: the image's ``flash_attn`` distribution is a
  namespace directory holding only the FA4 CuteDSL python.

So TokenSpeed has had no FlashAttention path for MHA on this card, while vLLM
0.26 serves the same model on the same GPU on FlashAttention-2. This module
closes that gap using the same binary vLLM uses.

How FA2 runs on sm_120 at all
-----------------------------
It is **PTX JIT, not native SASS**. vLLM's fork clamps the FA2 target list to
``8.0+PTX`` unconditionally::

    # vllm-flash-attn CMakeLists.txt
    cuda_archs_loose_intersection(FA2_ARCHS "8.0+PTX" "${CUDA_ARCHS}")

``cuobjdump --list-elf`` on the shipped ``_vllm_fa2_C.abi3.so`` returns 52
``sm_80`` cubins and ``--list-ptx`` returns 52 ``compute_80`` PTX images, on
every vLLM build inspected (cu129 and cu130). Building for ``12.0`` changes
nothing. So what runs here is Ampere-generation FlashAttention-2 —
``mma.m16n8k16``, no TMA, no warp specialisation — JIT-compiled by the driver
for sm_120 at first launch.

Two consequences worth stating plainly, because neither is visible in a
steady-state benchmark:

1. **First launch is expensive.** Measured on a cold JIT cache in this image:
   31 s for the first extend shape and 53 s for the first decode shape. The
   driver caches the result (``CUDA_CACHE_PATH``, default ``~/.nv``), so it is
   paid once per (kernel, arch, driver) and never again — but in a container
   with no persistent ``~/.nv`` it is paid on every cold start.
2. **There is no sm_120-tuned FA2 anywhere.** Nobody ships one; the sgl-kernel
   fork that does emit ``sm_120a`` FA2 SASS covers only bf16 head_dim 64/128
   split kernels, which this model's head_dim 256 excludes. Any future gain
   here is a compile-target change, not a wiring change.

Where the binary comes from
---------------------------
``vllm_flash_attn`` is not on PyPI in any usable form — the abandoned
``vllm-flash-attn`` 2.6.2 is built for torch 2.4/cu121, and ``flash-attn``
2.8.3 is sdist-only. The working artifact is the extension shipped inside the
vLLM container images. ``vllm/vllm-openai:cu130-nightly`` is
torch 2.11.0+cu130 / Python 3.12 / cxx11-ABI, an exact match for
``tokenspeed:sm120-dev``, and its ``_vllm_fa2_C.abi3.so`` loads unmodified.
(The ``v0.26.0-cu129`` image's copy does **not**: it links ``libcudart.so.12``
and this image has only CUDA 13.)

This module deliberately does not vendor or dlopen a path. It imports
``vllm_flash_attn`` (or ``vllm.vllm_flash_attn``) and registers nothing if the
import fails, so packaging stays an image concern.

Padded ``q``
------------
Under CUDA-graph bucketing ``q.shape[0]`` may exceed the live token count.
Detecting the real count would mean reading ``cu_seqlens_q[-1]`` on the host —
a device-to-host sync on every layer of every step.

FA2's varlen entry point does not need it. Its grid is
``(cdiv(max_seqlen_q, block_m), batch, num_heads)`` and each block clips
against ``cu_seqlens_q``, so rows past ``cu_seqlens_q[batch]`` are never
written and never read. Verified directly: with 0, 7 and 64 padding rows
appended to a ``batch=4, q_len=32, ctx=4096`` call, the real rows are
**bit-identical** to the unpadded call. So ``q`` is handed over whole and the
padding tail of the output is left as ``torch.empty_like`` produced it — the
same convention ``flashinfer_paged_mha_extend`` follows, for the same reason.

Declared traits
---------------
Narrow on purpose. FA2 does implement sliding windows and logit softcapping
natively, and this wrapper could forward both in two lines — but neither is
verified here, and a kernel that claims a feature it has not proven wins
selection and returns wrong answers. They stay ``False`` and selection keeps
routing those requests to the in-tree Triton kernel. Attention sinks are a
genuine gap: ``flash_attn_varlen_func`` raises ``NotImplementedError`` for
``s_aux`` on FA2.

Known cost
----------
``mha_decode_with_kvcache`` has to synthesise a ``cu_seqlens_q`` that the
decode dispatch does not pass. It is a pure function of
``(batch, max_seqlen_q)``, so it is built once per bucket and cached; the cache
is keyed on values, never on data pointers, and the tensor it returns is
constant, which makes it safe to bake into a captured graph.
"""

from __future__ import annotations

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)

# Inside the plugin band (16-19), FA2 prefill and extend outrank the FlashInfer
# paged wrappers at 16. The offset is a measurement, not a preference -- see
# kda-sm120/docs/log.md E33. Extend: 3.0-13.7x faster than the FlashInfer extend
# kernel *as it is registered today*, because that one calls plan() and rebuilds
# the CSR page index on every call of every layer. Prefill: 3.2-3.3x faster than
# the in-tree Triton kernel, which is its only competitor.
_FA2_PRIORITY_OFFSET = 1

# Decode is OFF by default, and that is the measurement too. Kernel for kernel
# FA2 loses decode to FlashInfer's paged prefill at every shape swept:
#
#   shape                     FA2 vs FlashInfer run()   FA2 vs in-tree Triton
#   b8  ctx4096 q_len 1       1.7x slower               3.0x faster
#   b32 ctx4096 q_len 1       1.0x  (tie)               1.0x  (tie)
#   b64 ctx2048 q_len 1       1.2x slower               1.2x slower
#   b8  ctx4096 q_len 4       5.9x slower               1.6x slower
#   b32 ctx4096 q_len 4       3.6x slower               2.4x slower
#
# The q_len 4 column is structural, not tuning. FA2's varlen paged entry point
# enables split-KV only when ``max_seqlen_q == 1`` -- ``flash_api.cpp``
# ``mha_varlen_fwd`` gates ``set_params_splitkv`` on ``seqlenq_ngroups_swapped``
# under the comment "Only apply split-k for decoding", and the else-branch
# hard-checks ``num_splits <= 1, "num_splits > 1 is not supported for varlen
# paged KV"``. So a speculative draft block gets no KV parallelism at all and
# wastes all but 4 rows of a 64-row query tile. It cannot be forced on.
#
# Unlike extend, the decode comparison is not distorted by plan() overhead: the
# runtime hoists the FlashInfer decode plan out of the captured graph (E27), so
# run()-only is what production actually pays. FA2 decode is therefore kept
# reachable -- it needs no plan, no prepare hook and nothing that is illegal
# under CUDA-graph capture, which is worth having if FlashInfer is ever absent
# -- but it does not win selection unless someone asks for it.
_REGISTER_DECODE = os.environ.get("UNIEINFRA_FA2_REGISTER_DECODE", "0") == "1"


def _load_fa2():
    """Return ``flash_attn_varlen_func`` from whichever vLLM layout is present.

    vLLM moved the extension under the ``vllm`` package during 0.26; a
    standalone repack keeps it top-level. Both are accepted so that this module
    does not dictate how the image installs it.

    Returns:
        The ``flash_attn_varlen_func`` callable.

    Raises:
        ImportError: If no layout is importable, or FA2 itself is unavailable
            inside an otherwise importable package (a ROCm or FA3-only build).
    """
    last: Exception | None = None
    for mod_name in ("vllm_flash_attn", "vllm.vllm_flash_attn"):
        try:
            mod = __import__(mod_name, fromlist=["flash_attn_varlen_func"])
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            last = exc
            continue
        if not getattr(mod, "FA2_AVAILABLE", True):
            last = ImportError(
                f"{mod_name} imported but FA2 is unavailable: "
                f"{getattr(mod, 'FA2_UNAVAILABLE_REASON', 'unknown')}"
            )
            continue
        return mod.flash_attn_varlen_func
    raise ImportError(
        "neither vllm_flash_attn nor vllm.vllm_flash_attn provides FA2"
    ) from last


# Cache of the synthetic decode qo_indptr, keyed on values only. A pointer key
# would be unsafe here for the opposite of the usual reason: the tensor is
# ours, and two different buckets must never share one.
_cu_seqlens_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}


def _uniform_cu_seqlens(
    batch: int, seqlen_q: int, device: torch.device
) -> torch.Tensor:
    """``[0, q, 2q, ..., batch*q]`` as int32, built once per bucket.

    Decode packs a fixed number of query tokens per request, so the cumulative
    length array is a constant of the bucket. Rebuilding it per call would add
    an allocation and a launch to every attention layer of every step.

    Args:
        batch: Number of requests.
        seqlen_q: Query tokens per request (1 for decode, draft length for
            speculative decode).
        device: Device to build on.

    Returns:
        int32 tensor of shape ``[batch + 1]``.
    """
    key = (device, batch, seqlen_q)
    cached = _cu_seqlens_cache.get(key)
    if cached is None:
        cached = torch.arange(batch + 1, dtype=torch.int32, device=device) * seqlen_q
        _cu_seqlens_cache[key] = cached
    return cached


def _unsupported(window_left: int, logit_cap: float, sinks) -> str | None:
    """Name the first declared-unsupported option that was passed, if any.

    Selection is supposed to make this unreachable — every one of these is
    declared ``False`` in the registration. It is checked anyway because the
    failure mode of the alternative is silence: FA2 would ignore the option and
    return a plausible, wrong tensor.
    """
    if window_left >= 0:
        return f"window_left={window_left} (sliding window is not registered)"
    if logit_cap != 0.0:
        return f"logit_cap={logit_cap} (logit softcap is not registered)"
    if sinks is not None:
        return "sinks (FA2 has no s_aux path)"
    return None


def fa2_mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MHA extend against a paged KV cache, on FlashAttention-2.

    TokenSpeed's extend contract is FA2's varlen-paged contract almost exactly:
    ``cu_seqlens_q`` is FA2's ``cu_seqlens_q``, ``page_table`` is its
    ``block_table``, and ``cache_seqlens`` is its ``seqused_k``. Nothing is
    gathered, reshaped or copied on the way in.

    ``cu_seqlens_kv`` is unused. FA2 takes the visible KV length per request
    from ``seqused_k``; passing both a ``cu_seqlens_k`` and a ``seqused_k`` is
    rejected by the wrapper.

    Args:
        q: Queries, ``[total_q, num_q_heads, head_dim]``. Rows beyond
            ``cu_seqlens_q[batch]`` are CUDA-graph padding and are neither read
            nor written; see the module docstring.
        cu_seqlens_q: Query cumulative lengths, ``[batch + 1]``, int32.
        cu_seqlens_kv: Unused.
        k_cache: Paged keys, ``[num_pages, page_size, num_kv_heads, head_dim]``.
        v_cache: Paged values, same shape as ``k_cache``.
        page_table: Page ids per sequence, ``[batch, max_pages_per_seq]``.
        cache_seqlens: Visible KV length per sequence, ``[batch]``.
        max_seqlen_q: Longest query in the batch.
        max_seqlen_k: Longest visible KV length in the batch.
        is_causal: Whether queries are a causal suffix of the cached KV. FA2's
            causal mask is bottom-right aligned, which is the same convention.
        window_left: Sliding window. Only -1 is registered.
        logit_cap: Logit softcap. Only 0.0 is registered.
        sinks: Attention sinks. Only None is registered.
        return_lse: Whether to also return log-sum-exp,
            ``[total_q, num_q_heads]``.
        softmax_scale: QK scale; defaults to ``1/sqrt(head_dim)``.
        q_scale: Unused — the block-scaled path is not registered.
        k_scale: Unused.
        v_scale: Unused.

    Returns:
        The attention output shaped like ``q``, or ``(out, lse)`` when
        ``return_lse``.

    Raises:
        NotImplementedError: If an option this kernel declares it does not
            support was passed anyway.
    """
    del cu_seqlens_kv, q_scale, k_scale, v_scale
    bad = _unsupported(window_left, logit_cap, sinks)
    if bad is not None:
        raise NotImplementedError(f"fa2_mha_extend_with_kvcache: {bad}")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    out = _flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=cache_seqlens,
        block_table=page_table,
        softmax_scale=softmax_scale,
        causal=is_causal,
        return_softmax_lse=return_lse,
        fa_version=2,
    )
    if return_lse:
        # FA2 returns lse as [num_heads, total_q]; the operator contract is
        # [total_q, num_heads].
        return out[0], out[1].transpose(0, 1)
    return out


def fa2_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MHA decode against a paged KV cache, on the same FA2 varlen entry point.

    Decode is extend with a uniform query length, so it goes through the same
    kernel rather than a separate one — the unification vLLM's FA2 backend also
    uses. The only thing decode does not hand us is a ``cu_seqlens_q``, and for
    a uniform query length that is a constant of the bucket; see
    :func:`_uniform_cu_seqlens`.

    Unlike the FlashInfer path this needs no ``plan()``, so there is no
    host-side scheduling, nothing that is illegal under CUDA-graph capture, and
    no prepare hook for the runtime to call. FA2 decides its split-KV factor on
    device from ``seqused_k``.

    Args:
        q: Queries, ``[batch * max_seqlen_q, num_q_heads, head_dim]``. Trailing
            CUDA-graph padding rows are permitted and are left untouched.
        k_cache: Paged keys, ``[num_pages, page_size, num_kv_heads, head_dim]``.
        v_cache: Paged values, same shape as ``k_cache``.
        page_table: Page ids per sequence, ``[batch, max_pages_per_seq]``.
        cache_seqlens: Visible KV length per sequence including the tokens
            being decoded now, ``[batch]``.
        max_seqlen_k: Longest visible KV length.
        max_seqlen_q: Query tokens per request — 1 for ordinary decode, the
            draft length under speculative decoding.
        window_left: Sliding window. Only -1 is registered.
        logit_cap: Logit softcap. Only 0.0 is registered.
        sinks: Attention sinks. Only None is registered.
        return_lse: Whether to also return log-sum-exp.
        softmax_scale: QK scale; defaults to ``1/sqrt(head_dim)``.
        q_scale: Unused — the block-scaled path is not registered.
        k_scale: Unused.
        v_scale: Unused.

    Returns:
        The attention output shaped like ``q``, or ``(out, lse)``.

    Raises:
        NotImplementedError: If an option this kernel declares it does not
            support was passed anyway.
    """
    del q_scale, k_scale, v_scale
    bad = _unsupported(window_left, logit_cap, sinks)
    if bad is not None:
        raise NotImplementedError(f"fa2_mha_decode_with_kvcache: {bad}")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    batch = int(cache_seqlens.shape[0])
    cu_seqlens_q = _uniform_cu_seqlens(batch, int(max_seqlen_q), q.device)

    out = _flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=cache_seqlens,
        block_table=page_table,
        softmax_scale=softmax_scale,
        # Bottom-right aligned, so a single query row attends the whole cache
        # either way; it only bites for speculative draft blocks.
        causal=max_seqlen_q > 1,
        return_softmax_lse=return_lse,
        fa_version=2,
    )
    if return_lse:
        return out[0], out[1].transpose(0, 1)
    return out


def fa2_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Unpaged causal MHA prefill on FA2's varlen entry point.

    The contiguous-KV case: no page table, and ``cu_seqlens`` indexes both q
    and kv.

    Args:
        q: Queries, ``[total, num_q_heads, head_dim]``.
        k: Keys, ``[total, num_kv_heads, head_dim]``.
        v: Values, same shape as ``k``.
        cu_seqlens: Cumulative lengths, ``[batch + 1]``, int32, shared by q
            and kv.
        cu_seqlens_cpu: Host mirror; unused, FA2 needs no host copy.
        max_seqlen: Longest sequence in the batch.
        window_left: Sliding window. Only -1 is registered.
        logit_cap: Logit softcap. Only 0.0 is registered.
        sinks: Attention sinks. Only None is registered.
        return_lse: Whether to also return log-sum-exp.
        softmax_scale: QK scale; defaults to ``1/sqrt(head_dim)``.
        q_scale: Unused.
        k_scale: Unused.
        v_scale: Unused.

    Returns:
        The attention output shaped like ``q``, or ``(out, lse)``.

    Raises:
        NotImplementedError: If an option this kernel declares it does not
            support was passed anyway.
    """
    del cu_seqlens_cpu, q_scale, k_scale, v_scale
    bad = _unsupported(window_left, logit_cap, sinks)
    if bad is not None:
        raise NotImplementedError(f"fa2_mha_prefill: {bad}")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    out = _flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_k=max_seqlen,
        cu_seqlens_k=cu_seqlens,
        softmax_scale=softmax_scale,
        causal=True,
        return_softmax_lse=return_lse,
        fa_version=2,
    )
    if return_lse:
        return out[0], out[1].transpose(0, 1)
    return out


# Bound by register_flash_attn2(); left None so that importing this module on a
# machine without the extension is harmless.
_flash_attn_varlen_func = None


def register_flash_attn2() -> None:
    """Register the FA2 MHA kernels, or explain why they are absent.

    Never raises. Discovery imports every plugin in-process, so an exception
    here would take the engine down over a missing optional wheel.
    """
    global _flash_attn_varlen_func

    from tokenspeed_kernel.platform import (
        ArchVersion,
        CapabilityRequirement,
        current_platform,
    )
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures

    platform = current_platform()
    if not (platform.is_nvidia and platform.arch_version == ArchVersion(12, 0)):
        logger.info(
            "unieinfra-kernel-sm120: not registering FA2, platform is %s sm_%s%s",
            platform.vendor,
            platform.arch_version.major,
            platform.arch_version.minor,
        )
        return

    try:
        _flash_attn_varlen_func = _load_fa2()
    except Exception as exc:  # noqa: BLE001 - a missing wheel is not a crash
        logger.warning(
            "unieinfra-kernel-sm120: FA2 kernels unavailable (%s: %s). Install "
            "the vllm_flash_attn extension from a vLLM image whose torch build "
            "matches this one (torch 2.11.0+cu130 -> vllm/vllm-openai:"
            "cu130-nightly); selection falls back to FlashInfer or Triton.",
            type(exc).__name__,
            exc,
        )
        return

    priority = int(Priority.PLUGIN) + _FA2_PRIORITY_OFFSET
    capability = CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        # A max is set for the reason the skill gives: the capability gate is
        # version-monotonic, so without one a future arch inherits a kernel
        # nobody tested there. It matters more than usual here, because what
        # actually executes is JIT-compiled Ampere PTX and its performance on
        # an unseen arch is unknowable from this side.
        max_arch_version=ArchVersion(12, 9),
    )
    signatures = format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    )
    # Everything below that is a bare False means "not implemented and not
    # verified here", never "off". Selection routes such requests to the
    # in-tree Triton kernel.
    traits = {
        "sliding_window": frozenset({False}),
        "support_sinks": frozenset({False}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
    }

    register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="unieinfra_fa2_mha_extend_sm120",
        solution="fa2",
        capability=capability,
        signatures=signatures,
        priority=priority,
        traits={"is_causal": frozenset({False, True}), **traits},
        tags={"unieinfra", "sm120"},
    )(fa2_mha_extend_with_kvcache)

    if _REGISTER_DECODE:
        register_kernel(
            "attention",
            "mha_decode_with_kvcache",
            name="unieinfra_fa2_mha_decode_sm120",
            solution="fa2",
            capability=capability,
            signatures=signatures,
            priority=priority,
            traits=traits,
            tags={"unieinfra", "sm120"},
        )(fa2_mha_decode_with_kvcache)

    register_kernel(
        "attention",
        "mha_prefill",
        name="unieinfra_fa2_mha_prefill_sm120",
        solution="fa2",
        capability=capability,
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=priority,
        traits=traits,
        tags={"unieinfra", "sm120"},
    )(fa2_mha_prefill)

    logger.info(
        "unieinfra-kernel-sm120: registered FA2 MHA prefill and extend at "
        "priority %d; decode %s (set UNIEINFRA_FA2_REGISTER_DECODE=1 to "
        "enable, but FlashInfer's kernel is faster at every decode shape "
        "measured). PTX-JIT from compute_80: the first launch of each shape "
        "costs tens of seconds on a cold CUDA JIT cache.",
        priority,
        "registered" if _REGISTER_DECODE else "NOT registered",
    )
