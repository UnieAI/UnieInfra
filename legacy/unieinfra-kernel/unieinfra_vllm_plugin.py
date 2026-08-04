# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""The ``vllm.general_plugins`` entry point and the out-of-tree quant config.

See :mod:`unieinfra_kernel.integrations.vllm` for why this seam was chosen and
how it reaches every dense GEMM.

**This module lives at the top level on purpose.** vLLM invokes the entry point
in *every* process it starts, including runs that never ask for our kernels. If
the entry point pointed inside :mod:`unieinfra_kernel`, merely loading it would
run the package ``__init__`` and therefore full kernel discovery (which imports
Triton). Living outside the package, it imports vLLM and nothing else until a
model actually requests ``--quantization unieinfra`` — so a shared box running
stock-vLLM benchmarks is not perturbed by our presence.

Everything here is additive: with the package uninstalled, or with the
``--quantization unieinfra`` flag absent, vLLM behaves exactly as it does
upstream.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["register", "UnieInfraConfig", "UnieInfraLinearMethod"]

# Quantization name the user passes to `--quantization`. It is a dispatch
# selector, not a numeric format: weights stay in the checkpoint dtype.
QUANT_NAME = "unieinfra"

# Default: offer EVERY dense Linear to the registry. Confinement is the
# registry's job, not a hand-maintained layer list -- `m_le_8` / `n_le_16384`
# already exclude prefill and wide projections, and the wrapper's runtime
# profitability rule declines anything else, delegating to `F.linear`. Measured
# on Qwen3-8B/sm_120 (see docs/experiments-log.md E34): routing all dense
# Linears is +3.9% batch-1 decode vs stock, against +1.6% for a hand-picked
# `qkv_proj,o_proj` list -- the difference is `down_proj`, which a manual list
# would have missed.
_DEFAULT_LAYERS = "all"

_registered = False


def _target_layers() -> tuple[str, ...]:
    """Layer-prefix substrings that opt in to registry dispatch."""
    raw = os.environ.get("UNIEINFRA_VLLM_LAYERS", _DEFAULT_LAYERS).strip()
    if raw.lower() in ("all", "*"):
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def register() -> None:
    """vLLM ``general_plugins`` entry point: install the UnieInfra kernel layer.

    Called once per vLLM process (front end, EngineCore, Worker) from
    ``vllm.plugins.load_general_plugins``. It registers the ``unieinfra``
    quantization config so that ``--quantization unieinfra`` routes dense Linear
    GEMMs through :func:`unieinfra_kernel.dispatch.linear_op`.

    Kernel **discovery is deferred** to :func:`discover_kernels`, which the
    config class calls the first time vLLM actually instantiates it. That keeps
    the plugin free for every vLLM run in the same environment that does not ask
    for it: not one ``unieinfra_kernel`` module is imported, no kernel is
    registered, and no dispatch decision changes. Registering a quantization
    *name* has no effect until it is requested.

    Failures are logged and swallowed: vLLM's own plugin loader is lenient
    (``vllm/plugins/__init__.py``), and a broken kernel package must never stop
    an engine from starting.
    """
    global _registered
    if _registered:
        return

    try:
        from vllm.model_executor.layers.quantization import (
            register_quantization_config,
        )

        config_cls, _method_cls = _lazy.resolve()
        register_quantization_config(QUANT_NAME)(config_cls)
        _registered = True
        logger.debug(
            "unieinfra_kernel: registered '%s' quantization config "
            "(kernel discovery deferred until requested)",
            QUANT_NAME,
        )
        if os.environ.get("UNIEINFRA_VLLM_EAGER_DISCOVERY", "0") != "0":
            discover_kernels()
    except Exception:  # noqa: BLE001 - never break engine startup
        logger.exception("unieinfra_kernel: vLLM plugin registration failed")


def discover_kernels() -> None:
    """Import :mod:`unieinfra_kernel` so every first-party kernel self-registers.

    Idempotent. Logs the resolved registry backend and the full kernel roster,
    which is the startup-side evidence that our selection layer is live inside
    the vLLM process.
    """
    import unieinfra_kernel  # noqa: F401  (side effect: kernel discovery)
    from unieinfra_kernel.boundary import BACKEND, KernelRegistry

    registry = KernelRegistry.get()
    logger.info(
        "unieinfra_kernel: selection layer active — backend=%s, "
        "%d kernel(s) across %d operator(s): %s",
        BACKEND,
        len(registry.list_kernels()),
        len(registry.list_operators()),
        ", ".join(sorted(spec.name for spec in registry.list_kernels())),
    )


_linear_op = None


def _get_linear_op():
    """Return :func:`unieinfra_kernel.dispatch.linear_op`, importing on first use.

    Importing anything under ``unieinfra_kernel`` runs the package ``__init__``,
    which is what performs kernel discovery — so this is also the point at which
    the registry is guaranteed populated.
    """
    global _linear_op
    if _linear_op is None:
        discover_kernels()
        from unieinfra_kernel.dispatch import linear_op

        _linear_op = linear_op
    return _linear_op


# ---------------------------------------------------------------------------
# Quantized modes
# ---------------------------------------------------------------------------
# `--quantization unieinfra` is a *dispatch selector*, not a numeric format. On
# a quantized checkpoint it therefore delegates every bit of the checkpoint path
# — parameter creation, loading, post-load processing, MoE, KV-cache scales — to
# the vLLM quantization config that owns the format, and overrides exactly one
# method: `LinearMethodBase.apply`. Two consequences worth stating:
#
#   * The A/B against stock vLLM is exact. Both arms hold *the same* weight
#     tensors produced by *the same* loader; only the GEMM differs.
#   * When our registry has no kernel for a shape, dispatch calls the engine's
#     own quantized GEMM, so a declined shape is bit-identical to stock vLLM.

# Mode names this plugin understands.
MODE_DENSE = "dense"
MODE_FP8 = "fp8"
MODE_NVFP4 = "nvfp4"

_fallbacks_registered = False


def _requested_mode() -> str:
    """Read ``UNIEINFRA_VLLM_QUANT``: ``auto`` (default), ``dense``, ``fp8``, ``nvfp4``."""
    raw = os.environ.get("UNIEINFRA_VLLM_QUANT", "auto").strip().lower()
    if raw in ("", "none", "off"):
        return MODE_DENSE
    if raw in ("auto", MODE_DENSE, MODE_FP8, MODE_NVFP4):
        return raw
    logger.warning(
        "unieinfra_kernel: ignoring UNIEINFRA_VLLM_QUANT=%r "
        "(expected auto/dense/fp8/nvfp4)",
        raw,
    )
    return "auto"


def _detect_checkpoint_mode(hf_quant_config: dict[str, Any] | None) -> str | None:
    """Infer the checkpoint's weight format from its HF ``quantization_config``.

    Args:
        hf_quant_config: The checkpoint's quantization config dict, or ``None``
            for an unquantized checkpoint.

    Returns:
        ``"nvfp4"``, ``"fp8"``, or ``None`` when the checkpoint is not quantized
        in a format this plugin routes.
    """
    if not hf_quant_config:
        return None
    algo = str(
        hf_quant_config.get("quant_algo")
        or hf_quant_config.get("quantization", {}).get("quant_algo", "")
        or ""
    ).upper()
    method = str(hf_quant_config.get("quant_method", "")).lower()
    if "NVFP4" in algo or "FP4" in algo:
        return MODE_NVFP4
    if "FP8" in algo or "fp8" in method:
        return MODE_FP8
    return None


def _register_engine_fallbacks() -> None:
    """Hand vLLM's own quantized GEMMs to the dispatcher as the fallback path.

    ``unieinfra_kernel`` never imports vLLM (AGENTS.md: the kernel package stays
    engine-neutral), so the engine injects its GEMMs here. These are the *same*
    calls vLLM's own linear kernels make, which is what makes "no kernel matched"
    equivalent to running stock vLLM for that shape.
    """
    global _fallbacks_registered
    if _fallbacks_registered:
        return

    import torch as _torch
    from unieinfra_kernel.dispatch import register_engine_fallback
    from vllm import _custom_ops as vllm_ops

    def _fp8_engine_gemm(
        x: "torch.Tensor",
        weight: "torch.Tensor",
        weight_scale: "torch.Tensor",
        bias: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """vLLM's W8A8 fp8 path: dynamic per-token quant + CUTLASS scaled mm."""
        qx, x_scale = vllm_ops.scaled_fp8_quant(
            x, scale=None, use_per_token_if_dynamic=True
        )
        return vllm_ops.cutlass_scaled_mm(
            qx,
            weight,
            scale_a=x_scale,
            scale_b=weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )

    def _nvfp4_engine_gemm(
        x: "torch.Tensor",
        weight: "torch.Tensor",
        weight_scale_swizzled: "torch.Tensor",
        input_global_scale_inv: "torch.Tensor",
        alpha: "torch.Tensor",
        bias: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """vLLM's W4A4 NVFP4 path: block-quantize the activation + CUTLASS FP4 mm."""
        x_fp4, x_blockscale = vllm_ops.scaled_fp4_quant(
            x,
            input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="cutlass",
            padded_n=x.shape[-1],
        )
        out = vllm_ops.cutlass_scaled_fp4_mm(
            x_fp4, weight, x_blockscale, weight_scale_swizzled, alpha, x.dtype
        )
        if bias is not None:
            out = out + bias
        return out

    del _torch
    register_engine_fallback(MODE_FP8, _fp8_engine_gemm)
    register_engine_fallback(MODE_NVFP4, _nvfp4_engine_gemm)
    _fallbacks_registered = True


class _Fp8Adapter:
    """Reads a vLLM fp8 Linear layer and calls the fp8 dispatch op.

    vLLM keeps the fp8 weight transposed (``[K, N]``, so ``.t()`` is the
    contiguous ``[N, K]`` a streaming kernel wants) with a per-tensor or
    per-channel fp32 scale. Nothing here re-lays-out the checkpoint.
    """

    kind = MODE_FP8

    def __init__(self) -> None:
        from unieinfra_kernel.dispatch import layer_is_claimable, linear_fp8_op

        # Resolved once, at construction: a deferred import inside ``apply`` is
        # a hard Dynamo graph break (E34).
        self._op = linear_fp8_op
        self._claimable = layer_is_claimable

    def prepare(self, layer: "torch.nn.Module") -> None:
        """Nothing to stash before vLLM processes the weights."""

    def claim(self, layer: "torch.nn.Module") -> bool:
        """Whether the post-load layer state is one this adapter can serve."""
        weight = getattr(layer, "weight", None)
        scale = getattr(layer, "weight_scale", None)
        if weight is None or scale is None:
            return False
        if weight.dtype not in (torch.float8_e4m3fn,):
            return False
        if weight.dim() != 2 or weight.stride(0) != 1:
            return False
        if getattr(layer, "input_scale", None) is not None:
            # Static activation scales mean the engine expects a pre-quantized
            # activation; leave that path entirely to vLLM.
            return False
        n = weight.shape[1]
        if scale.numel() not in (1, n):
            return False
        # Ask the registry whether *any* batch size could select one of our
        # kernels for this projection. If not, leave the layer entirely alone:
        # wrapping it would add the dispatch op's Inductor-opaque cost for no
        # possible benefit.
        return self._claimable("fp8", n, weight.shape[0], torch.bfloat16)

    def apply(
        self,
        layer: "torch.nn.Module",
        x: "torch.Tensor",
        bias: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """Route the layer's GEMM through fp8 kernel selection."""
        return self._op(x, layer.weight, layer.weight_scale, bias)


class _NvFp4Adapter:
    """Reads a vLLM ModelOpt NVFP4 Linear layer and calls the nvfp4 dispatch op.

    One extra tensor is kept: the block scales in **linear** layout. vLLM
    swizzles them into the CUTLASS 128x4 tiling during post-load processing,
    which a streaming kernel cannot index cheaply. The copy costs ``N x K / 16``
    bytes (1/8 of the packed weight) and buys a layout both paths can use — the
    swizzled tensor stays exactly as vLLM built it, so the engine fallback is
    untouched.
    """

    kind = MODE_NVFP4

    def __init__(self) -> None:
        from unieinfra_kernel.dispatch import layer_is_claimable, linear_nvfp4_op

        self._op = linear_nvfp4_op
        self._claimable = layer_is_claimable

    def prepare(self, layer: "torch.nn.Module") -> None:
        """Snapshot the un-swizzled block scales before vLLM rewrites them.

        Only for projections the registry could actually serve. The copy is
        ``N x K / 16`` bytes; skipping the ones we can never claim (Qwen3-8B's
        ``gate_up_proj`` is the largest of them) halves the extra footprint.
        """
        scale = getattr(layer, "weight_scale", None)
        weight = getattr(layer, "weight", None)
        if scale is None or weight is None or scale.dim() != 2:
            return
        if not self._claimable(
            "nvfp4", weight.shape[0], weight.shape[1] * 2, torch.bfloat16
        ):
            return
        layer.unieinfra_weight_scale_linear = scale.data.clone()

    def claim(self, layer: "torch.nn.Module") -> bool:
        """Whether the post-load layer state is one this adapter can serve."""
        raw = getattr(layer, "unieinfra_weight_scale_linear", None)
        weight = getattr(layer, "weight", None)
        if raw is None or weight is None:
            return False
        if weight.dtype != torch.uint8 or weight.dim() != 2:
            return False
        # CUTLASS padding rewrites the weight; a padded layer is left to vLLM.
        if getattr(layer, "weights_padding_cols", 0):
            return False
        if weight.shape[0] != getattr(layer, "output_size_per_partition", -1):
            return False
        if raw.shape != (weight.shape[0], weight.shape[1] * 2 // 16):
            return False
        if not all(
            getattr(layer, name, None) is not None
            for name in ("alpha", "weight_global_scale", "input_global_scale_inv")
        ):
            return False
        # See `_Fp8Adapter.claim`: a projection the registry can never serve is
        # left unwrapped rather than paying the dispatch op's cost for nothing.
        return self._claimable(
            "nvfp4", weight.shape[0], weight.shape[1] * 2, torch.bfloat16
        )

    def apply(
        self,
        layer: "torch.nn.Module",
        x: "torch.Tensor",
        bias: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """Route the layer's GEMM through nvfp4 kernel selection."""
        return self._op(
            x,
            layer.weight,
            layer.unieinfra_weight_scale_linear,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.input_global_scale_inv,
            layer.alpha,
            bias,
        )


_adapters: dict[str, Any] = {}
_wrapped_classes: dict[tuple, type] = {}


def _get_adapter(mode: str):
    """Return (and memoise) the layer adapter for a quantized ``mode``."""
    adapter = _adapters.get(mode)
    if adapter is None:
        discover_kernels()
        _register_engine_fallbacks()
        adapter = {MODE_FP8: _Fp8Adapter, MODE_NVFP4: _NvFp4Adapter}[mode]()
        _adapters[mode] = adapter
    return adapter


def _wrapped_method_class(base_cls: type, mode: str) -> type:
    """Build a subclass of ``base_cls`` whose ``apply`` is registry-dispatched.

    Everything else — ``create_weights``, the weight loader, MoE handling,
    ``uses_meta_device`` — is inherited untouched, which is what keeps the
    checkpoint path identical to stock vLLM.

    Args:
        base_cls: The vLLM ``LinearMethodBase`` subclass that owns this format.
        mode: :data:`MODE_FP8` or :data:`MODE_NVFP4`.

    Returns:
        A cached subclass of ``base_cls``.
    """
    key = (base_cls, mode)
    cached = _wrapped_classes.get(key)
    if cached is not None:
        return cached

    adapter = _get_adapter(mode)

    class _UnieInfraScaledLinearMethod(base_cls):  # type: ignore[valid-type,misc]
        """vLLM's own quantized linear method with a registry-selected GEMM."""

        unieinfra_mode = mode

        def process_weights_after_loading(self, layer):
            """Run vLLM's post-load processing, then decide whether we claim it."""
            adapter.prepare(layer)
            super().process_weights_after_loading(layer)
            claimed = adapter.claim(layer)
            layer.unieinfra_claimed = claimed
            if not claimed:
                logger.debug(
                    "unieinfra_kernel: %s layer not claimable, leaving to vLLM",
                    mode,
                )

        def apply(self, layer, x, bias=None):
            """Compute the GEMM through the registry, or vLLM's own path.

            ``layer.unieinfra_claimed`` is a plain Python bool set at load time,
            so Dynamo folds this branch at trace time — no graph break.
            """
            if not getattr(layer, "unieinfra_claimed", False):
                return super().apply(layer, x, bias)
            return adapter.apply(layer, x, bias)

    # KEEP THE BASE CLASS NAME. ``LinearBase.__init__`` picks the modern
    # (``BasevLLMParameter``) weight loader with
    # ``self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED``
    # (``vllm/model_executor/layers/linear.py:473`` and ``:1618``) — a *string*
    # membership test, not an ``isinstance`` check. Renaming the subclass drops
    # every quantized layer onto the legacy loader, which then tries to copy a
    # packed ``[N, K//2]`` uint8 shard into an unpacked parameter and dies on
    # ``assert param_data.shape == loaded_weight.shape``. Our identity lives in
    # ``unieinfra_mode`` instead.
    _UnieInfraScaledLinearMethod.__name__ = base_cls.__name__
    _UnieInfraScaledLinearMethod.__qualname__ = base_cls.__qualname__
    _wrapped_classes[key] = _UnieInfraScaledLinearMethod
    return _UnieInfraScaledLinearMethod


def _build_delegate(mode: str, hf_quant_config: dict[str, Any] | None):
    """Construct the vLLM quantization config that owns ``mode``'s checkpoint.

    Args:
        mode: :data:`MODE_FP8` or :data:`MODE_NVFP4`.
        hf_quant_config: The checkpoint's quantization config dict, if any.

    Returns:
        A vLLM ``QuantizationConfig`` instance.
    """
    if mode == MODE_NVFP4:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        if not hf_quant_config:
            raise ValueError(
                "UNIEINFRA_VLLM_QUANT=nvfp4 needs a ModelOpt NVFP4 checkpoint; "
                "this model has no quantization_config."
            )
        return ModelOptNvFp4Config.from_config(hf_quant_config)

    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if (
        hf_quant_config
        and "fp8" in str(hf_quant_config.get("quant_method", "")).lower()
    ):
        return Fp8Config.from_config(hf_quant_config)
    # Unquantized checkpoint: vLLM's online per-tensor fp8, quantized during
    # loading — the same thing `--quantization fp8` gives stock vLLM.
    return Fp8Config()


def _build_config_class():
    """Define :class:`UnieInfraConfig` lazily against vLLM's base classes."""
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    class _UnieInfraLinearMethod(UnquantizedLinearMethod):
        """``LinearMethodBase`` whose ``apply()`` is registry-selected.

        Weight creation and loading are inherited verbatim from
        ``UnquantizedLinearMethod`` — this is a dispatch override, not a
        quantization scheme, so the checkpoint path is untouched.
        """

        def __init__(self, prefix: str = "") -> None:
            super().__init__()
            self.prefix = prefix
            # Resolve at construction, never inside ``apply``: ``apply`` is
            # traced by Dynamo, and a deferred import or a log call there is a
            # hard graph break (observed: gb0291, "logging method call").
            self._linear_op = _get_linear_op()

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Compute the layer's GEMM through the capability registry.

            Args:
                layer: The vLLM Linear layer (supplies ``layer.weight``).
                x: Activations ``[num_tokens, K]``.
                bias: Optional ``[N]`` bias.

            Returns:
                ``[num_tokens, N]`` output tensor.
            """
            return self._linear_op(x, layer.weight, bias)

    class _UnieInfraConfig(QuantizationConfig):
        """Out-of-tree ``QuantizationConfig`` that swaps the GEMM, not the dtype.

        ``get_quant_method`` dispatches per layer *instance* and can therefore
        see the full state-dict ``prefix``, which is how the decode GEMV is
        confined to ``qkv_proj`` / ``o_proj`` without touching MLP or LM-head
        shapes.
        """

        def __init__(
            self,
            targets: tuple[str, ...] = (),
            mode: str | None = None,
            delegate: Any = None,
        ) -> None:
            # Both set before ``super().__init__()``, which assigns through the
            # ``packed_modules_mapping`` property below.
            self.delegate = delegate
            self._packed_modules_mapping: dict = {}
            super().__init__()
            discover_kernels()
            if mode is None:
                # Reached through ``quant_cls()`` — an unquantized checkpoint.
                # ``auto`` means dense there; an explicit env value still wins,
                # which is how online fp8 is requested from a bf16 checkpoint.
                requested = _requested_mode()
                mode = MODE_DENSE if requested == "auto" else requested
            self.targets = targets or _target_layers()
            self.mode = mode
            if delegate is None and mode != MODE_DENSE:
                # `quant_cls()` (no HF quantization_config) reached us with a
                # mode forced by the environment: build the delegate now.
                self.delegate = _build_delegate(mode, None)
                self.delegate.packed_modules_mapping = self._packed_modules_mapping
            logger.info(
                "unieinfra_kernel: quantization mode=%s (delegate=%s)",
                mode,
                type(self.delegate).__name__ if self.delegate else "none",
            )

        # `packed_modules_mapping` is written by the *model* at construction
        # (`model_loader/utils.py`), and the delegate's exclusion logic reads it.
        # Forwarding keeps a single dict so both see the same thing.
        @property
        def packed_modules_mapping(self) -> dict:
            """The model's fused-module map, shared with the delegate."""
            if self.delegate is not None:
                return self.delegate.packed_modules_mapping
            return self._packed_modules_mapping

        @packed_modules_mapping.setter
        def packed_modules_mapping(self, value: dict) -> None:
            self._packed_modules_mapping = value
            if self.delegate is not None:
                self.delegate.packed_modules_mapping = value

        def get_name(self) -> str:
            """Name this config is registered under."""
            return QUANT_NAME

        def get_supported_act_dtypes(self) -> list[torch.dtype]:
            """Activation dtypes the registered kernels cover."""
            if self.delegate is not None:
                return self.delegate.get_supported_act_dtypes()
            return [torch.float16, torch.bfloat16, torch.float32]

        @classmethod
        def override_quantization_method(
            cls,
            hf_quant_cfg: dict[str, Any],
            user_quant: str | None,
            hf_config: Any = None,
        ) -> str | None:
            """Claim a *pre-quantized* checkpoint when the user asked for us.

            Without this, ``ModelConfig._verify_quantization`` refuses to run a
            checkpoint whose ``quantization_config`` says ``modelopt`` under
            ``--quantization unieinfra``. It returns ``None`` for every other
            request, so no run that did not ask for us is affected.

            Args:
                hf_quant_cfg: The checkpoint's quantization config dict.
                user_quant: The method the user passed on the command line.
                hf_config: The HF model config (unused).

            Returns:
                ``"unieinfra"`` when the user selected us, else ``None``.
            """
            del hf_quant_cfg, hf_config
            return QUANT_NAME if user_quant == QUANT_NAME else None

        def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
            """Forward vLLM's weight-name remapping to the delegate."""
            if self.delegate is not None:
                self.delegate.apply_vllm_mapper(hf_to_vllm_mapper)

        def maybe_update_config(self, model_name: str, hf_config: Any = None) -> None:
            """Forward vLLM's late config fix-ups to the delegate."""
            if self.delegate is not None:
                self.delegate.maybe_update_config(model_name, hf_config=hf_config)

        def get_cache_scale_mapper(self):
            """Forward the KV-cache scale name mapping to the delegate.

            ModelOpt checkpoints ship FP8 KV-cache scales; the delegate is the
            thing that knows how to name them (E30 lost a load to exactly this).
            """
            if self.delegate is not None:
                return self.delegate.get_cache_scale_mapper()
            return QuantizationConfig.get_cache_scale_mapper()

        @classmethod
        def get_min_capability(cls) -> int:
            """Minimum SM capability. Per-kernel capability gating is the registry's job."""
            return 80

        @staticmethod
        def get_config_filenames() -> list[str]:
            """No on-disk config: this is a runtime dispatch selector."""
            return []

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "_UnieInfraConfig":
            """Build from the checkpoint's HF ``quantization_config`` block.

            The block is what tells us the *weight format on disk*. An absent or
            unrecognised block means a dense bf16/fp16 checkpoint and the E34
            behaviour; ``modelopt``/``NVFP4`` or an fp8 block selects the
            matching delegate. ``UNIEINFRA_VLLM_QUANT`` overrides the inference,
            which is how online fp8 is requested from a bf16 checkpoint.

            Args:
                config: The model's quantization config dict (may be empty).

            Returns:
                A configured ``_UnieInfraConfig``.
            """
            discover_kernels()
            requested = _requested_mode()
            detected = _detect_checkpoint_mode(config)
            mode = detected or MODE_DENSE if requested == "auto" else requested
            if mode != MODE_DENSE and detected is not None and mode != detected:
                logger.warning(
                    "unieinfra_kernel: UNIEINFRA_VLLM_QUANT=%s but the checkpoint "
                    "is %s; honouring the checkpoint.",
                    mode,
                    detected,
                )
                mode = detected
            delegate = None if mode == MODE_DENSE else _build_delegate(mode, config)
            return cls(targets=_target_layers(), mode=mode, delegate=delegate)

        def get_quant_method(self, layer: torch.nn.Module, prefix: str):
            """Return the GEMM method for ``layer``, or ``None`` to keep vLLM's.

            In dense mode this is E34's behaviour verbatim. In a quantized mode
            the *delegate* produces the method — so weight creation, loading and
            post-load processing are stock vLLM — and we only swap in a subclass
            whose ``apply`` is registry-dispatched.

            Args:
                layer: The layer being constructed.
                prefix: Its full state-dict prefix, e.g.
                    ``model.layers.0.self_attn.qkv_proj``.

            Returns:
                A quantization method for this layer, or ``None`` when vLLM
                should keep its own default.
            """
            if self.delegate is not None:
                method = self.delegate.get_quant_method(layer, prefix)
                if method is None or not isinstance(layer, LinearBase):
                    # KV-cache, MoE, embeddings, excluded layers: untouched.
                    return method
                if os.environ.get("UNIEINFRA_VLLM_DISABLE", "0") != "0":
                    return method
                targets = self.targets or _target_layers()
                if targets and not any(target in prefix for target in targets):
                    return method
                wrapped = _wrapped_method_class(type(method), self.mode)
                # Re-class in place: every attribute the delegate's constructor
                # set (kernel objects, quant keys, marlin flags) is preserved,
                # and only ``apply`` / ``process_weights_after_loading`` change.
                method.__class__ = wrapped
                logger.debug(
                    "unieinfra_kernel: claiming %s linear layer %s", self.mode, prefix
                )
                return method

            if not isinstance(layer, LinearBase):
                # Embeddings, attention, MoE: vLLM substitutes its own default
                # when we return None.
                return None
            if os.environ.get("UNIEINFRA_VLLM_DISABLE", "0") != "0":
                return UnquantizedLinearMethod()
            targets = self.targets or _target_layers()
            if targets and not any(target in prefix for target in targets):
                return UnquantizedLinearMethod()
            logger.debug("unieinfra_kernel: claiming linear layer %s", prefix)
            return _UnieInfraLinearMethod(prefix=prefix)

    return _UnieInfraConfig, _UnieInfraLinearMethod


class _Lazy:
    """Descriptor-free lazy holder so importing this module needs no vLLM."""

    def __init__(self) -> None:
        self._config = None
        self._method = None

    def resolve(self):
        if self._config is None:
            self._config, self._method = _build_config_class()
        return self._config, self._method


_lazy = _Lazy()


def __getattr__(name: str):
    """Materialise ``UnieInfraConfig`` / ``UnieInfraLinearMethod`` on first use."""
    if name == "UnieInfraConfig":
        return _lazy.resolve()[0]
    if name == "UnieInfraLinearMethod":
        return _lazy.resolve()[1]
    raise AttributeError(name)
