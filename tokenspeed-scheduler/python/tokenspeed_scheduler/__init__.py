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

"""Python API for the TokenSpeed scheduler."""

import importlib.machinery
import importlib.util
import os
import sys

_EXT_MODULE = f"{__name__}.tokenspeed_scheduler_ext"


# An optional second extension, built TOKENSPEED_FLAT_KVCACHE=ON, installed
# beside the default one. The two builds are not interchangeable: the flat
# build compiles the radix insert/publish path out and its Scheduler
# constructor refuses an empty paged-cache-group list, while the radix build
# never populates flat block tables. So which .so a process loads is a
# property of the MODEL it is about to serve, not of the installation, and
# `tokenspeed.flat_kvcache_select` decides it from the checkpoint's
# architectures before anything imports this package.
#
# Both extensions define PyInit_tokenspeed_scheduler_ext, so the alternate one
# has to be loaded by path under the canonical module name rather than
# imported by a different one.
#
# Search every entry of __path__, not just __file__'s directory: an editable
# install splits this package across the worktree (this .py) and
# site-packages (the compiled .so), so the two do not live together.
def _find_flat_ext():
    for root in list(__path__) + [os.path.dirname(__file__)]:
        candidate = os.path.join(root, "flat", "ext.so")
        if os.path.exists(candidate):
            return candidate
    return None


def _load_ext():
    want_flat = os.environ.get(
        "TOKENSPEED_FLAT_KVCACHE_RUNTIME", ""
    ).strip().lower() not in ("", "0", "false", "off", "no")
    flat_path = _find_flat_ext() if want_flat else None
    if flat_path is not None:
        loader = importlib.machinery.ExtensionFileLoader(_EXT_MODULE, flat_path)
        spec = importlib.util.spec_from_loader(_EXT_MODULE, loader)
        module = importlib.util.module_from_spec(spec)
        # Register before exec_module: nanobind's own type registration looks
        # the module up by name while it is still initialising.
        sys.modules[_EXT_MODULE] = module
        spec.loader.exec_module(module)
        return module
    import tokenspeed_scheduler.tokenspeed_scheduler_ext as module

    return module


_ext = _load_ext()

DisaggregationMode = _ext.DisaggregationMode
ExecutionEvent = _ext.ExecutionEvent
ExecutionPlan = _ext.ExecutionPlan
FlatTerminalError = _ext.FlatTerminalError
FlatTerminalReason = _ext.FlatTerminalReason
PagedCacheGroupAllocator = _ext.PagedCacheGroupAllocator
PagedCacheGroupConfig = _ext.PagedCacheGroupConfig
PagedCacheGroupFamily = _ext.PagedCacheGroupFamily
PagedCacheGroupTable = _ext.PagedCacheGroupTable
PagedCacheRetention = _ext.PagedCacheRetention
PagedCacheTransferPolicy = _ext.PagedCacheTransferPolicy
PrefixCacheAdjunctSpec = _ext.PrefixCacheAdjunctSpec
RequestSpec = _ext.RequestSpec
Scheduler = _ext.Scheduler
SchedulerConfig = _ext.SchedulerConfig
SchedulerStats = _ext.SchedulerStats

PD = _ext.PD
Cache = _ext.Cache
Forward = _ext.Forward
ForwardEvent = _ext.ForwardEvent
KVEvent = _ext.KVEvent

# True iff the extension was compiled with TOKENSPEED_FLAT_KVCACHE (flat
# KvCacheCoordinator scheduler); False for the default radix build. Older
# extensions predate the attribute — default False, which matches their radix
# behavior.
FLAT_KVCACHE: bool = bool(getattr(_ext, "FLAT_KVCACHE", False))


def _flat_forward_op_repr(self):
    return (
        f"FlatForwardOp("
        f"request_ids={list(self.request_ids)}, "
        f"request_pool_indices={list(self.request_pool_indices)}, "
        f"input_lengths={list(self.input_lengths)}, "
        f"occupied_pages={[list(p) for p in self.occupied_pages]}, "
        f"begins={list(self.begins)}, "
        f"sizes={list(self.sizes)}, "
        f"input_ids={list(self.input_ids)}, "
        f"shifted_input_ids={list(self.shifted_input_ids)}, "
        f"extend_prefix_lens={list(self.extend_prefix_lens)}, "
        f"num_extends={self.num_extends()}"
        f")"
    )


Forward.FlatForwardOp.__repr__ = _flat_forward_op_repr

__all__ = [
    # Core
    "Scheduler",
    "SchedulerConfig",
    "SchedulerStats",
    "DisaggregationMode",
    "RequestSpec",
    "PagedCacheRetention",
    "PagedCacheGroupConfig",
    "PagedCacheGroupAllocator",
    "PagedCacheGroupFamily",
    "PagedCacheTransferPolicy",
    "PagedCacheGroupTable",
    "PrefixCacheAdjunctSpec",
    "FLAT_KVCACHE",
    # Execution plan & operations
    "ExecutionPlan",
    "FlatTerminalError",
    "FlatTerminalReason",
    "Forward",
    "PD",
    "Cache",
    "KVEvent",
    # Events
    "ExecutionEvent",
    "ForwardEvent",
]
