# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""UnieInfra kernel plugin for stock vLLM.

Load path
---------
``pyproject.toml`` advertises ``unieinfra_vllm_plugin:register`` on vLLM's
``vllm.general_plugins`` entry-point group, which vLLM invokes in *every*
process — front end, EngineCore and Worker — before ``VllmConfig`` is built and
long before the model is constructed. So a plain ``pip install
unieinfra-kernel`` is enough; no vLLM source is touched and nothing is forked.

The implementation deliberately sits in the top-level :mod:`unieinfra_vllm_plugin`
module rather than in this package, so that loading the entry point costs a vLLM
run that does not want us exactly nothing (see that module's docstring).

Seam
----
The GEMM is reached through :func:`register_quantization_config`, not through
``CustomOp.register_oot`` / ``PluggableLayer``. That choice dodges two traps
that silently break the layer-replacement routes (both verified in the installed
vLLM 0.25.1):

1. ``vllm/config/vllm.py:1182-1189`` appends ``"none"`` to ``custom_ops``
   whenever the Inductor backend is on and compilation is not ``NONE``, so a
   third-party ``forward_cuda`` override is never dispatched. Irrelevant here:
   a ``LinearMethodBase`` has no ``enabled()`` gate.
2. ``PluggableLayer.__new__`` keys the out-of-tree swap on ``cls.__name__``
   (``vllm/model_executor/custom_op.py:49``), and only ``ReplicatedLinear``
   (:291), ``ColumnParallelLinear`` (:396) and ``RowParallelLinear`` (:1536) are
   registered — ``QKVParallelLinear`` (:942) and ``MergedColumnParallelLinear``
   (:580) are unregistered *subclasses*, so replacing ``column_parallel_linear``
   would miss ``qkv_proj``, which is the exact shape we are here to accelerate.
   Irrelevant here: ``get_quant_method`` is called per layer *instance*
   (``linear.py:274-279``), so subclasses are covered by construction.

The quantization seam reaches every dense Linear because the returned method's
``apply()`` is the GEMM: ``linear.py:402`` (``ReplicatedLinear``), ``:597``
(``ColumnParallelLinear``, inherited by ``QKVParallelLinear`` and
``MergedColumnParallelLinear``) and ``:1763`` (``RowParallelLinear``).

Usage
-----
::

    vllm serve /models/Qwen3-8B --quantization unieinfra

``unieinfra`` is a *dispatch* config, not a numeric format. On a bf16 checkpoint
the model stays bf16 and weight creation/loading are inherited unchanged from
``UnquantizedLinearMethod``, so the checkpoint path is byte-identical to stock.

Quantized checkpoints
---------------------
On an fp8 or ModelOpt-NVFP4 checkpoint the config **delegates** to vLLM's own
quantization config for that format (``Fp8Config`` / ``ModelOptNvFp4Config``) and
re-classes the ``LinearMethodBase`` it produces into a subclass that overrides
only ``apply`` and ``process_weights_after_loading``. Parameter creation, the
weight loader, MoE and the KV-cache method stay the delegate's, so both arms of a
comparison hold identical weights and only the GEMM differs::

    vllm serve nvidia/Qwen3-8B-NVFP4 --quantization unieinfra \\
        --linear-backend cutlass --attention-backend TRITON_ATTN

The subclass **keeps the base class's ``__name__``**: ``LinearBase.__init__``
selects the modern weight loader with a *string* membership test on
``self.quant_method.__class__.__name__`` (``linear.py:473``, ``:1618``), so a
renamed subclass silently falls back to the legacy loader and dies loading a
packed uint8 shard.

When the registry has no kernel for a shape, dispatch calls the engine GEMM the
plugin injected through ``unieinfra_kernel.dispatch.register_engine_fallback``
— ``unieinfra_kernel`` never imports vLLM — so a declined shape behaves exactly
as stock vLLM would.

Environment
-----------
``UNIEINFRA_VLLM_QUANT``
    ``auto`` (default; read from the checkpoint), ``dense``, ``fp8`` or
    ``nvfp4``. ``fp8`` on a bf16 checkpoint gives vLLM's online per-tensor fp8.
``UNIEINFRA_VLLM_LAYERS``
    Comma-separated substrings of the layer prefix that opt in. Default ``all``;
    the registry's trait gate does the real confinement.
``UNIEINFRA_KERNEL_LOG_SELECTION=1``
    Log every distinct ``(M, N, K) -> kernel`` decision.
``UNIEINFRA_KERNEL_SELECTION_DUMP=<prefix>``
    Write ``<prefix>.<pid>.json`` with the selection record and the kernel call
    counters. Selection happens in vLLM's *worker* process, whose logging config
    does not carry our logger, so this is how the decision is proven.
``UNIEINFRA_VLLM_DISABLE=1``
    Load the plugin but never claim a layer (A/B control arm).
"""

from __future__ import annotations

from unieinfra_vllm_plugin import register

__all__ = ["register"]
