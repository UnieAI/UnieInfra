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

"""AMD GPU fabric detection from the amdkfd sysfs topology."""

from pathlib import Path

import pytest

from tokenspeed.runtime.distributed.amd_fabric import (
    kfd_gpus_pcie_only,
    local_gpus_pcie_only,
)

_PCIE, _XGMI = 2, 11


def _write_topology(root: Path, cpus: int, gpus: int, gpu_link_type: int) -> Path:
    """Write a minimal amdkfd topology: CPU nodes first, then GPU nodes."""
    nodes = root / "nodes"
    gpu_ids = list(range(cpus, cpus + gpus))
    for node in range(cpus + gpus):
        node_dir = nodes / str(node)
        node_dir.mkdir(parents=True)
        version = 90500 if node in gpu_ids else 0
        (node_dir / "properties").write_text(
            f"cpu_cores_count 8\ngfx_target_version {version}\n"
        )
        if node not in gpu_ids:
            continue
        io_link = node_dir / "io_links" / "0"
        io_link.mkdir(parents=True)
        (io_link / "properties").write_text(f"type {_PCIE}\nnode_to 0\n")
        for index, peer in enumerate(p for p in gpu_ids if p != node):
            link = node_dir / "p2p_links" / str(index)
            link.mkdir(parents=True)
            (link / "properties").write_text(
                f"type {gpu_link_type}\nversion_major 0\nnode_to {peer}\n"
            )
    return nodes


def test_pcie_only_gpus_are_detected(tmp_path):
    nodes = _write_topology(tmp_path, cpus=2, gpus=4, gpu_link_type=_PCIE)
    assert kfd_gpus_pcie_only(nodes)


def test_xgmi_connected_gpus_keep_symmetric_memory(tmp_path):
    nodes = _write_topology(tmp_path, cpus=2, gpus=8, gpu_link_type=_XGMI)
    assert not kfd_gpus_pcie_only(nodes)


def test_unreadable_or_single_gpu_topology_keeps_defaults(tmp_path):
    assert not kfd_gpus_pcie_only(tmp_path / "missing")
    nodes = _write_topology(tmp_path, cpus=1, gpus=1, gpu_link_type=_PCIE)
    assert not kfd_gpus_pcie_only(nodes)


@pytest.mark.parametrize(("fabric", "pcie_only"), [("xgmi", False), ("pcie", True)])
def test_env_override_forces_the_fabric(monkeypatch, fabric, pcie_only):
    monkeypatch.setenv("TOKENSPEED_AMD_GPU_FABRIC", fabric)
    assert local_gpus_pcie_only() is pcie_only


def test_env_override_rejects_unknown_fabric(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_AMD_GPU_FABRIC", "infiniband")
    with pytest.raises(ValueError, match="TOKENSPEED_AMD_GPU_FABRIC"):
        local_gpus_pcie_only()
