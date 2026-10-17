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

"""Out-of-tree kernels for NVIDIA sm_120 (RTX PRO 6000 Blackwell).

tokenspeed-kernel's registry reserves priority band 16-19 for plugins so they
can outrank in-tree defaults without the tree being patched. Everything here
uses that band, and nothing here modifies tokenspeed-kernel.

The registrations declare narrow traits on purpose. A kernel that claims a
feature it does not implement wins selection and then produces wrong output,
which is worse than not being selected — so sliding windows, logit caps,
attention sinks and the block-scaled path are all left to the in-tree Triton
kernel until this plugin implements them.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register() -> None:
    """Register this plugin's kernels. Called by ``discover_plugins()``.

    Registration is skipped, with a log line rather than an exception, on any
    platform this plugin was not written for. A plugin that raises during
    discovery takes the host process down with it.
    """
    import torch
    from tokenspeed_kernel.platform import (
        ArchVersion,
        CapabilityRequirement,
        current_platform,
    )
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures
    from unieinfra_kernel_sm120.flashinfer_mha import flashinfer_paged_mha_decode

    platform = current_platform()
    if not (platform.is_nvidia and platform.arch_version == ArchVersion(12, 0)):
        logger.info(
            "unieinfra-kernel-sm120: not registering, platform is %s sm_%s%s",
            platform.vendor,
            platform.arch_version.major,
            platform.arch_version.minor,
        )
        return

    # The extend kernel is deliberately NOT registered. Measured in the live
    # engine (E36) it is a net loss: FlashInfer's plan() memcpys leave the GPU
    # idle 1.449 ms per step, giving 653 tok/s against 666 without it (-2.6%).
    # The in-tree Triton extend it would displace is also faster at this
    # geometry — 2.944 ms across the 16 full-attention layers, against
    # FlashAttention-2's 3.198 ms in vLLM.
    #
    # It is correct (cosine 0.99999678 against Triton, padded queries included)
    # and `flashinfer_paged_mha_extend` stays importable so the plan-hoist work
    # survives for whoever revisits it. Correct and slower is still a
    # regression, and priority 16 would have made it the default.

    register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="unieinfra_flashinfer_paged_mha_decode_sm120",
        solution="flashinfer_paged",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(12, 0),
            max_arch_version=ArchVersion(12, 9),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.PLUGIN,
        traits={
            "sliding_window": frozenset({False}),
            "sinks": frozenset({False}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
        tags={"unieinfra", "sm120"},
    )(flashinfer_paged_mha_decode)

    logger.info(
        "unieinfra-kernel-sm120: registered FlashInfer paged MHA decode at "
        "priority %d (extend deliberately unregistered, see E36)",
        int(Priority.PLUGIN),
    )

    # The remaining families live in their own modules so that separate work can
    # proceed on each without collisions. Each is isolated: a module that fails
    # to import or register leaves the others, and the engine, working. Silence
    # here would be its own bug — a kernel that is written, tested and never
    # reached is indistinguishable from one that was never written.
    for module_name, entry_point in (
        ("unieinfra_kernel_sm120.flash_attn2", "register_flash_attn2"),
        ("unieinfra_kernel_sm120.gdn", "register_gdn"),
        ("unieinfra_kernel_sm120.gemm", "register_gemm"),
        ("unieinfra_kernel_sm120.moe", "register_moe"),
        ("unieinfra_kernel_sm120.quant", "register_quant"),
    ):
        try:
            module = importlib.import_module(module_name)
            getattr(module, entry_point)()
        except Exception:
            logger.warning(
                "unieinfra-kernel-sm120: %s.%s failed; its kernels are "
                "unavailable and selection will fall back in tree",
                module_name,
                entry_point,
                exc_info=True,
            )
