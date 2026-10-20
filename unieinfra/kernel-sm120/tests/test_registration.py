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

"""Every kernel the plugin means to register on sm_12x is registered.

``register()`` isolates each family module and logs a failure instead of
raising, so a broken family looks the same as a family that was never written:
the engine just falls back in tree. This is the check that tells them apart.
"""

from __future__ import annotations

import pytest
import torch

# FA2 needs a vllm_flash_attn extension the image may not carry; gemm and moe
# are deliberately unregistered on main (see register()).
EXPECTED = {
    "unieinfra_flashinfer_paged_mha_decode_sm120",
    "unieinfra_gdn_chunk_prefill_sm120",
    "unieinfra_flashinfer_quantize_mxfp4_sm120",
    "unieinfra_flashinfer_quantize_nvfp4_sm120",
    "unieinfra_triton_quantize_fp8_with_scale_sm120",
    "unieinfra_triton_quantize_fp8_sm120",
}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12,
    reason="requires an sm_12x device (sm_120 or sm_121)",
)


def test_plugin_registers_every_expected_kernel():
    from tokenspeed_kernel.plugins import discover_plugins, list_plugins

    discover_plugins()
    plugins = {p.name: p for p in list_plugins()}
    assert "unieinfra_sm120" in plugins
    missing = EXPECTED - set(plugins["unieinfra_sm120"].kernel_names)
    assert not missing, f"registered nowhere: {sorted(missing)}"
