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

import pytest
import torch
from tokenspeed_kernel import hadamard_transform

torch.manual_seed(42)


@pytest.mark.parametrize("solution", ["triton", "fast_hadamard_transform"])
def test_hadamard_transform(device: str, solution: str, require) -> None:
    dtype = torch.bfloat16
    require("transform", "hadamard_transform", solution, dtype, "x")

    x = torch.randn((3, 5, 128), device=device, dtype=dtype)
    scale = 128**-0.5

    out = hadamard_transform(x, scale=scale, solution=solution)

    expected = x.float().reshape(-1, x.shape[-1])
    h = 1
    while h < expected.shape[-1]:
        expected_3d = expected.reshape(
            expected.shape[0], expected.shape[1] // (2 * h), 2 * h
        )
        left = expected_3d[..., :h]
        right = expected_3d[..., h : 2 * h]
        combined = torch.cat((left + right, left - right), dim=-1)
        expected = combined.reshape_as(expected)
        h *= 2
    expected = expected.reshape_as(x) * scale

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(
        out.float(),
        expected.to(dtype).float(),
        atol=2.0e-2,
        rtol=2.0e-2,
    )


def test_is_writable_dir_accepts_a_creatable_path(tmp_path) -> None:
    from tokenspeed_kernel.ops.transform import weak_ref

    assert weak_ref._is_writable_dir(str(tmp_path / "new" / "deep"))
    assert (tmp_path / "new" / "deep").is_dir()
    # The probe file must not be left behind.
    assert list((tmp_path / "new" / "deep").iterdir()) == []


def test_is_writable_dir_rejects_a_path_under_a_file(tmp_path) -> None:
    """A candidate below a regular file is rejected for any uid.

    Root ignores permission bits, so the unwritable case is expressed as a
    path that cannot be a directory at all rather than a mode-restricted one.
    """
    from tokenspeed_kernel.ops.transform import weak_ref

    blocker = tmp_path / "blocker"
    blocker.write_text("")
    assert not weak_ref._is_writable_dir(str(blocker / "child"))


def test_resolve_build_directory_prefers_the_tokenspeed_override(
    tmp_path, monkeypatch
) -> None:
    from tokenspeed_kernel.ops.transform import weak_ref

    monkeypatch.setenv("TOKENSPEED_EXTENSIONS_DIR", str(tmp_path / "ts"))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "torch"))
    assert weak_ref._resolve_build_directory() == str(tmp_path / "ts")


def test_resolve_build_directory_falls_back_to_a_temp_dir(
    tmp_path, monkeypatch
) -> None:
    """The read-only-$HOME case: every env/`$HOME` candidate is unusable.

    This is the container condition that used to degrade ``weak_ref_tensor``
    to the identity fallback, which made breakable-graph capture memory scale
    with the bucket sum and forced the prefill graph onto a single rung.
    """
    from tokenspeed_kernel.ops.transform import weak_ref

    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setenv("TOKENSPEED_EXTENSIONS_DIR", str(blocker / "ts"))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(blocker / "torch"))
    monkeypatch.setenv("HOME", str(blocker / "home"))
    tmp_root = tmp_path / "tmproot"
    tmp_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_root))
    monkeypatch.setattr(weak_ref.tempfile, "tempdir", None)

    resolved = weak_ref._resolve_build_directory()

    assert resolved is not None
    assert resolved.startswith(str(tmp_root))
