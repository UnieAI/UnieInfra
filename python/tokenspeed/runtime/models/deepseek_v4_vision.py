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

"""DeepSeek V4 vision tower (image encoder for V4-Flash-Vision-Exp).

Faithful port of the checkpoint's own ``inference/vision.py``, extended to the
weights actually shipped: ``attention.{wqkv,wo}``, ``patch_embed.proj`` and
``aligner.{w1,w2}`` all carry a bias (the upstream reference omits it) while
the SwiGLU ``mlp`` does not. The ``Aligner`` flattens a ``downsample_ratio`` x
``downsample_ratio`` patch neighborhood through a GELU MLP projecting
``vision_dim * ratio**2 -> hidden_size``.

Tower parameters live under ``vision.*`` (``patch_embed``, ``blocks``,
``norm``) and ``aligner.*`` so the engine's generic weight loader picks them up
unaligned from a ``DeepseekV4ForCausalLM`` that registers ``self.vision`` and
``self.aligner``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping

# Sentinel token slots inside an image block (see the checkpoint's
# image_processor.py). ``IMG`` carries the real aligner tokens; the rest are
# learnable scaffold embeddings. ``vocab_size + kind`` is the token id.
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4


def llm_grid(n_vit_h: int, n_vit_w: int, ratio: int) -> tuple[int, int]:
    """Aligner-output grid height/width (N-layout) for ``n_vit`` patches."""
    return math.ceil(n_vit_h / ratio), math.ceil(n_vit_w / ratio)


def build_image_block(
    n_llm_h: int, n_llm_w: int, start_pos: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the ``(types, perm)`` pair the checkpoint's own preprocessor uses.

    ``types`` is the token-kind sequence (sentinel 0..4) in final block order,
    ``perm`` maps each block ``IMAGE`` slot to the aligner output row (in the
    pre-``perm`` grid order). Reconstructed from the grid size and the block's
    start position so the engine encoder does not depend on the gateway
    forwarding extra tensors.
    """
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def _patch_area(config: PretrainedConfig) -> int:
    return 3 * int(config.vision_patch_size) ** 2


@lru_cache(16)
def _get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class VisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV4VisionPatchEmbed(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.proj = nn.Linear(_patch_area(config), config.vision_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either ``(N, 3, p, p)`` patch tiles (reference layout) or the
        # already-flattened ``(N, 3*p*p)`` a preprocessor may hand over.
        if x.dim() == 4:
            x = x.flatten(1)
        return self.proj(x)


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.rope_theta = config.vision_rope_theta
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim, bias=True)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim, bias=True)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        cos, sin = _get_vision_cos_sin(n_h, n_w, self.head_dim // 2, self.rope_theta)
        cos = cos.to(x.device)
        sin = sin.to(x.device)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.norm1 = VisionRMSNorm(config.vision_dim)
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = VisionRMSNorm(config.vision_dim)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), n_h, n_w)
        return x + self.mlp(self.norm2(x))


class DeepseekV4VisionAligner(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        lm_dim = getattr(config, "dim", None) or config.hidden_size
        self.w1 = nn.Linear(in_dim, lm_dim, bias=True)
        self.w2 = nn.Linear(lm_dim, lm_dim, bias=True)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


class DeepseekV4VisionTower(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE.

    Registered as ``self.vision`` on the causal LM; ``self.aligner`` is the
    matching ``DeepseekV4VisionAligner``. ``encode_image`` runs patch-embed ->
    32 blocks -> norm -> aligner and returns the downsampled text-width tokens.
    """

    def __init__(self, config: PretrainedConfig, mapping: Mapping | None = None):
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.patch_embed = DeepseekV4VisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            [DeepseekV4VisionBlock(config) for _ in range(config.vision_n_layers)]
        )
        self.norm = VisionRMSNorm(config.vision_dim)

    def encode_image(
        self,
        patches: torch.Tensor,
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        """``patches``: ``(n_h * n_w, 3, p, p)`` (or flattened). Returns the
        ``(downsampled_tokens, vision_dim)`` ViT output before the aligner."""
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, n_h, n_w)
        return self.norm(x)


class DeepseekV4VisionModel(nn.Module):
    """Full image encoder: vision tower + aligner, used by the encode path."""

    def __init__(self, config: PretrainedConfig, mapping: Mapping | None = None):
        super().__init__()
        self.vision = DeepseekV4VisionTower(config, mapping=mapping)
        self.aligner = DeepseekV4VisionAligner(config)

    def encode(
        self,
        patches: torch.Tensor,
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        """Return text-width image tokens: ``aligner(vit(patches, n_h, n_w))``."""
        x = self.vision.encode_image(patches, n_h, n_w)
        return self.aligner(x, n_h, n_w)
