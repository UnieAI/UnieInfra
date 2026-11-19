# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
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

"""NUMA-local CPU sets for AMD devices come from the PCI device's sysfs."""

import builtins
import io
from types import SimpleNamespace

from tokenspeed_kernel import platform


def test_parse_cpulist_ranges_and_singletons():
    assert platform._parse_cpulist("0-2,5,7-8\n") == (0, 1, 2, 5, 7, 8)
    assert platform._parse_cpulist("") == ()


def test_sysfs_affinity_follows_each_devices_pci_address(monkeypatch):
    props = {
        0: SimpleNamespace(pci_domain_id=0, pci_bus_id=0x73, pci_device_id=0),
        1: SimpleNamespace(pci_domain_id=0, pci_bus_id=0xA3, pci_device_id=0),
    }
    files = {
        "/sys/bus/pci/devices/0000:73:00.0/local_cpulist": "0-1,4\n",
        "/sys/bus/pci/devices/0000:a3:00.0/local_cpulist": "2-3\n",
    }
    monkeypatch.setattr(platform.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(platform.torch.cuda, "get_device_properties", props.get)
    real_open = builtins.open
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *a, **k: (
            io.StringIO(files[path]) if path in files else real_open(path, *a, **k)
        ),
    )
    assert platform._detect_sysfs_numa_cpu_affinity() == ((0, 1, 4), (2, 3))


def test_sysfs_affinity_is_empty_when_unreadable(monkeypatch):
    monkeypatch.setattr(platform.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        platform.torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(pci_domain_id=0, pci_bus_id=0xFF, pci_device_id=31),
    )
    assert platform._detect_sysfs_numa_cpu_affinity() == ()
