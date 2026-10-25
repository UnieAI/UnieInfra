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

"""Readiness probe for CUDA fabric-handle (MNNVL) memory.

Multi-node NVLink buffers need the full IMEX stack (nvidia-imex daemon plus
/dev/nvidia-caps-imex-channels), which only exists on NVLink-domain machines
such as GB200 NVL72. Machine-type, device-attribute and NVLS-multicast checks
are all insufficient: drivers report fabric-handle *support* and multicast
support on hosts where the IMEX stack is absent and the allocation itself fails
with CUDA_ERROR_NOT_PERMITTED. Every MNNVL consumer therefore has to probe the
allocation, so the probe lives here instead of being duplicated per consumer.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import socket
from collections.abc import Sequence
from pathlib import Path

import torch
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "fabric_allocation_supported",
    "gather_fabric_map",
    "group_has_fabric",
    "group_has_multicast",
    "group_host_span",
    "multicast_bind_supported",
]

logger = logging.getLogger(__name__)

# CUDA driver ABI constants used by the fabric probe (stable since CUDA 12).
_CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_HANDLE_TYPE_FABRIC = 8
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
_CU_MULTICAST_GRANULARITY_MINIMUM = 0

_probe_cache: dict[int, bool] = {}
_multicast_probe_cache: dict[int, bool] = {}
_fabric_map: list[bool] | None = None
_host_map: list[int] | None = None
_multicast_map: list[bool] | None = None


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocationPropAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmulticastObjectProp(ctypes.Structure):
    # Mirrors CUmulticastObjectProp_st from cuda.h.
    _fields_ = [
        ("numDevices", ctypes.c_uint),
        ("size", ctypes.c_size_t),
        ("handleTypes", ctypes.c_ulonglong),
        ("flags", ctypes.c_ulonglong),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    # Mirrors CUmemAllocationProp_st from cuda.h.
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocationPropAllocFlags),
    ]


def _probe_fabric_allocation(device_index: int) -> bool:
    """Run one granularity-sized fabric-handle ``cuMemCreate`` and release it."""
    for lib_name in ("libcuda.so.1", "libcuda.so"):
        try:
            libcuda = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    else:
        return False

    try:
        if libcuda.cuInit(0) != 0:
            return False
        device = ctypes.c_int()
        if libcuda.cuDeviceGet(ctypes.byref(device), device_index) != 0:
            return False
        supported = ctypes.c_int(0)
        if (
            libcuda.cuDeviceGetAttribute(
                ctypes.byref(supported),
                _CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                device,
            )
            != 0
            or supported.value == 0
        ):
            return False

        # VMM calls need a current context; reuse the caller's (torch usually
        # made one), else retain the device primary context and restore after.
        prev_ctx = ctypes.c_void_p()
        if libcuda.cuCtxGetCurrent(ctypes.byref(prev_ctx)) != 0:
            return False
        retained = False
        if not prev_ctx.value:
            primary_ctx = ctypes.c_void_p()
            if libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(primary_ctx), device) != 0:
                return False
            retained = True
            libcuda.cuCtxSetCurrent(primary_ctx)
        try:
            prop = _CUmemAllocationProp()
            prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
            prop.requestedHandleTypes = _CU_MEM_HANDLE_TYPE_FABRIC
            prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
            prop.location.id = device.value

            granularity = ctypes.c_size_t(0)
            if (
                libcuda.cuMemGetAllocationGranularity(
                    ctypes.byref(granularity),
                    ctypes.byref(prop),
                    _CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
                != 0
            ):
                return False

            handle = ctypes.c_ulonglong(0)
            if (
                libcuda.cuMemCreate(
                    ctypes.byref(handle),
                    ctypes.c_size_t(granularity.value),
                    ctypes.byref(prop),
                    ctypes.c_ulonglong(0),
                )
                != 0
            ):
                return False
            libcuda.cuMemRelease(handle)
            return True
        finally:
            if retained:
                libcuda.cuCtxSetCurrent(prev_ctx)
                libcuda.cuDevicePrimaryCtxRelease(device)
    except (OSError, AttributeError):
        return False


def fabric_allocation_supported(device_index: int) -> bool:
    """Return True iff the driver can serve fabric-handle memory on the device.

    Args:
        device_index: CUDA device ordinal to probe.

    Returns:
        Whether a fabric-handle allocation succeeds, i.e. whether MNNVL buffers
        can be created from this process. The result is cached per device
        because callers must all reach the same decision, and because the probe
        touches the driver.
    """
    cached = _probe_cache.get(device_index)
    if cached is None:
        cached = _probe_fabric_allocation(device_index)
        _probe_cache[device_index] = cached
        logger.info(f"fabric allocation on device {device_index!s}: {cached!s}")
    return cached


def _probe_multicast_bind(device_index: int) -> bool:
    """Bind one granularity of this device's memory into a two-device team.

    The driver reports multicast support from the hardware, so it stays set
    after the fabric manager loses track of the NVSwitch state (a fabric
    manager restart under live GPUs does it; the log says every GPU in the
    partition needs a reset). The bind is what fails, with
    CUDA_ERROR_ILLEGAL_STATE, and torch's symmetric memory then skips
    multicast and hands out a null multicast pointer that the multimem kernels
    store through. A one-device team is rejected outright, so the team takes
    a second visible device; binding only this device's memory needs nothing
    from the other.
    """
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    try:
        if libcuda.cuInit(0) != 0:
            return False
        count = ctypes.c_int(0)
        if libcuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return False
        if count.value < 2:
            # A single visible device has no peer to multicast to, and no
            # team to test with; leave the answer to the topology gates.
            return True
        device = ctypes.c_int()
        peer = ctypes.c_int()
        if libcuda.cuDeviceGet(ctypes.byref(device), device_index) != 0:
            return False
        if libcuda.cuDeviceGet(ctypes.byref(peer), 1 if device_index == 0 else 0) != 0:
            return False

        prev_ctx = ctypes.c_void_p()
        if libcuda.cuCtxGetCurrent(ctypes.byref(prev_ctx)) != 0:
            return False
        retained = False
        if not prev_ctx.value:
            primary_ctx = ctypes.c_void_p()
            if libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(primary_ctx), device) != 0:
                return False
            retained = True
            libcuda.cuCtxSetCurrent(primary_ctx)
        mc_handle = ctypes.c_ulonglong(0)
        mem_handle = ctypes.c_ulonglong(0)
        bound = False
        size = 0
        try:
            mc_prop = _CUmulticastObjectProp(
                2, 0, _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0
            )
            granularity = ctypes.c_size_t(0)
            if (
                libcuda.cuMulticastGetGranularity(
                    ctypes.byref(granularity),
                    ctypes.byref(mc_prop),
                    _CU_MULTICAST_GRANULARITY_MINIMUM,
                )
                != 0
            ):
                return False
            size = granularity.value
            mc_prop.size = size
            if (
                libcuda.cuMulticastCreate(
                    ctypes.byref(mc_handle), ctypes.byref(mc_prop)
                )
                != 0
            ):
                return False
            for member in (device, peer):
                if libcuda.cuMulticastAddDevice(mc_handle, member) != 0:
                    return False

            prop = _CUmemAllocationProp()
            prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
            prop.requestedHandleTypes = _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
            prop.location.id = device.value
            if (
                libcuda.cuMemCreate(
                    ctypes.byref(mem_handle),
                    ctypes.c_size_t(size),
                    ctypes.byref(prop),
                    ctypes.c_ulonglong(0),
                )
                != 0
            ):
                return False
            bound = (
                libcuda.cuMulticastBindMem(
                    mc_handle,
                    ctypes.c_size_t(0),
                    mem_handle,
                    ctypes.c_size_t(0),
                    ctypes.c_size_t(size),
                    ctypes.c_ulonglong(0),
                )
                == 0
            )
            return bound
        finally:
            if bound:
                libcuda.cuMulticastUnbind(
                    mc_handle, device, ctypes.c_size_t(0), ctypes.c_size_t(size)
                )
            if mem_handle.value:
                libcuda.cuMemRelease(mem_handle)
            if mc_handle.value:
                libcuda.cuMemRelease(mc_handle)
            if retained:
                libcuda.cuCtxSetCurrent(prev_ctx)
                libcuda.cuDevicePrimaryCtxRelease(device)
    except (OSError, AttributeError):
        return False


def multicast_bind_supported(device_index: int) -> bool:
    """Return True iff this device's memory binds into a multicast team.

    Cached per device. A False here on a host whose driver advertises
    multicast is a fabric fault, not a capability gap, so it warns: every
    multicast path on the host falls back to NCCL until the GPUs are reset.
    """
    cached = _multicast_probe_cache.get(device_index)
    if cached is None:
        cached = _probe_multicast_bind(device_index)
        _multicast_probe_cache[device_index] = cached
        if not cached:
            logger.warning(
                f"multicast bind failed on device {device_index!s}; "
                "multicast collectives fall back to NCCL"
            )
    return cached


def _host_identity() -> int:
    """An id every rank on one host computes identically, and no other host does.

    The boot id is the host's, not the container's: a launcher that gives each
    worker its own UTS namespace hands them all the same hostname, so hostnames
    would merge two machines into one and skip the fabric test that keeps a
    group off a rendezvous it cannot complete. Hostname is the last resort.
    """
    name = ""
    for path in ("/proc/sys/kernel/random/boot_id", "/etc/machine-id"):
        try:
            name = Path(path).read_text().strip()
        except OSError:
            continue
        if name:
            break
    if not name:
        name = socket.gethostname()
    digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
    # Signed 64-bit, because the gather carries it in an int64 tensor.
    return int.from_bytes(digest, "big") >> 1


def gather_fabric_map() -> list[bool]:
    """Gather and cache every world rank's fabric-allocation verdict and host.

    Fabric handles are an NVIDIA concept, so off NVIDIA the answer is no for
    every rank and is filled in without a collective. Deciding that here rather
    than at the call site keeps the collective out of a lazy path: a caller
    that skipped this on the wrong platform would otherwise trigger the gather
    from a gate, which is the dispatch-time collective this map exists to
    remove.

    That branch is the one thing here the ranks do not agree on by
    construction. ``is_nvidia`` is detected locally, so a job mixing CUDA and
    non-CUDA ranks would have some enter the all_gather and others return, and
    the ones that entered would wait out the NCCL timeout. The caller it
    replaced read a server argument, which was uniform; this reads the machine.
    """
    global _fabric_map, _host_map, _multicast_map

    if _fabric_map is not None:
        return _fabric_map

    world_size = torch.distributed.get_world_size()
    if not current_platform().is_nvidia:
        # No fabric to map, so every rank is its own host and every span declines.
        _fabric_map = [False] * world_size
        _host_map = list(range(world_size))
        _multicast_map = [False] * world_size
        return _fabric_map

    device = torch.device("cuda", torch.cuda.current_device())
    local = torch.tensor(
        [
            int(fabric_allocation_supported(device.index)),
            _host_identity(),
            int(multicast_bind_supported(device.index)),
        ],
        dtype=torch.int64,
        device=device,
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local, group=torch.distributed.group.WORLD)
    _fabric_map = [bool(value[0].item()) for value in gathered]
    _host_map = [int(value[1].item()) for value in gathered]
    _multicast_map = [bool(value[2].item()) for value in gathered]
    logger.info(
        f"fabric allocation available on {sum(_fabric_map)!s}/{len(_fabric_map)!s} "
        f"ranks across {len(set(_host_map))!s} hosts; multicast binds on "
        f"{sum(_multicast_map)!s}/{len(_multicast_map)!s}",
    )
    return _fabric_map


def group_has_fabric(ranks: Sequence[int]) -> bool:
    """Return whether every rank in ``ranks`` reported fabric support.

    Raises if the map has not been gathered, rather than gathering it here.
    This is asked from dispatch, where the ranks present are the group's and
    not the world's, so a lazy gather would run a world collective from a
    stage or a data-parallel subset and hang the ranks that never arrive. A
    missing map means the initialization hook did not run, which is a wiring
    bug; callers outside a server must gather it themselves.
    """
    if _fabric_map is None:
        raise RuntimeError(
            "fabric map was never gathered; call gather_fabric_map() at "
            "distributed initialization before any reachability gate"
        )
    return all(_fabric_map[rank] for rank in ranks)


def group_has_multicast(ranks: Sequence[int]) -> bool:
    """Return whether every rank in ``ranks`` bound memory into a multicast team.

    An ungathered map answers True: unlike the fabric verdict this is a fault
    check layered on the topology gates, and a caller outside a server that
    never gathered keeps the answer those gates gave before it existed.
    """
    if _multicast_map is None:
        return True
    return all(_multicast_map[rank] for rank in ranks)


def group_host_span(ranks: Sequence[int]) -> int | None:
    """How many distinct hosts ``ranks`` occupy, or ``None`` if never gathered.

    Placement is read from the map rather than divided out of the visible device
    count: a job running fewer workers than a host has GPUs puts two hosts inside
    one ``rank // device_count`` window, and a group called node-local that way
    skips the fabric test and hangs in the rendezvous instead of declining.

    ``None`` rather than a raise, because the caller is a gate that declines what
    it cannot establish. ``group_has_fabric`` raises because by the time it is
    asked, the decision to consult the map has already been taken.
    """
    if _host_map is None:
        return None
    return len({_host_map[rank] for rank in ranks})
