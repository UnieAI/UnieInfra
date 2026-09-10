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
"""The paged extend kernel must declare gemma-4's two prefill geometries.

gemma-4-31B-it is heterogeneous: 50 sliding layers at head_dim 256 and 10
global layers at ``global_head_dim`` 512 with 4 KV heads. The flashinfer paged
FA2 registration declared ``head_dim`` as ``{64, 128, 256}``, so selection
declined it for the ten global layers and they fell to the Triton fallback --
silently, because a closed trait set does not raise, it just does not match.

On the production shape (an 8192-token chunk against 35,326 cached tokens) that
cost 137.24 ms per layer at 122.2 TFLOP/s against flashinfer's 115.48 ms at
145.2. End to end on the 32-session shared-prefix corpus it was 8.8% to 11.6%
of the marginal per-prefill-token cost.

Runs in a subprocess. The registration is gated on an env var read when the ops
module is first imported, and `test/conftest.py` imports the package before any
test module, so setting it in-process is too late -- and forcing it with a
module reload mutates the process-wide registry, which made 124 unrelated tests
in later files fail with NoKernelFoundError. A child process gets the gate on
from birth and takes the shared state with it when it exits.
"""

import json
import subprocess
import sys

import pytest

_FI_PAGED = "flashinfer_paged_mha_extend_with_kvcache"

# (name, head_dim, sliding_window) for the two layer types gemma-4 prefills.
_GEMMA4_LAYERS = [
    ("sliding x50", 256, True),
    ("global x10", 512, False),
]

_PROBE = """
import json, os, sys
os.environ["TOKENSPEED_ENABLE_FI_PAGED_EXTEND"] = "1"
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels
from tokenspeed_kernel.selection import spec_matches_traits

load_builtin_kernels()
spec = KernelRegistry.get().get_by_name({name!r})
if spec is None:
    print(json.dumps({{"available": False}}))
    sys.exit(0)
traits = {{
    "head_dim": {head_dim},
    "is_causal": True,
    "sliding_window": {sliding},
    "support_sinks": False,
    "support_logit_cap": False,
    "return_lse": False,
}}
print(json.dumps({{"available": True, "matches": spec_matches_traits(spec, traits)}}))
"""


@pytest.mark.parametrize("label,head_dim,sliding", _GEMMA4_LAYERS)
def test_paged_extend_declares_gemma4_geometry(label, head_dim, sliding):
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE.format(name=_FI_PAGED, head_dim=head_dim, sliding=sliding),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, f"probe failed:\n{out.stderr[-2000:]}"
    result = json.loads(out.stdout.strip().splitlines()[-1])
    if not result["available"]:
        pytest.skip("flashinfer's paged prefill wrapper is not available")
    assert result["matches"], (
        f"gemma-4 {label} (head_dim {head_dim}) does not match {_FI_PAGED}; "
        f"those layers would fall to the Triton fallback."
    )
