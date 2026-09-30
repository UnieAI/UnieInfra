#!/usr/bin/env python3
"""staging 路徑的位元組層級往返測試：完全複製 _transfer 的分批與合併邏輯。

模型層級的比對分不出「搬錯位元組」和「前綴長度改變導致的數值差異」，兩者都
產生通順但不同的文字。這個測試不經過模型：把已知資料從裝置寫進檔案再讀回來，
逐位元組比對。staging 故意設得很小，強制走多批路徑。
"""

import os
import sys

import torch

sys.path.insert(0, "/workspace/roy/UnieInfra/tokenspeed-kernel/python")
sys.path.insert(0, "/workspace/roy/UnieInfra/python")
from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_ranges

from tokenspeed.runtime.cache.l2.storage import HostCacheStorage

PLANE = 524_288
N_PLANES = int(sys.argv[1]) if len(sys.argv) > 1 else 64
STAGE = int(sys.argv[2]) if len(sys.argv) > 2 else 16 << 20
PATH = "/workspace/roy/.rt.bin"

dev = torch.arange(N_PLANES * PLANE, dtype=torch.int64, device="cuda")
dev = (dev % 251).to(torch.uint8).contiguous()
stage = torch.empty(STAGE, dtype=torch.uint8, pin_memory=True)
view = memoryview(stage.numpy())
assert stage.data_ptr() % 4096 == 0

with open(PATH, "wb") as fh:
    os.posix_fallocate(fh.fileno(), 0, N_PLANES * PLANE)
fd = os.open(PATH, os.O_RDWR | os.O_DIRECT)
# 呼叫出貨的實作，不要複製一份會漂移的邏輯
io = object.__new__(HostCacheStorage)
io._fd = fd

# 打散順序，模擬真實的 range 分布（_transfer 會依檔案偏移重新排序）
order = list(range(N_PLANES))
order = order[1::2] + order[0::2]
ranges = [(0, i * PLANE, i * PLANE, PLANE) for i in order]


def run(direction, stream):
    ordered = sorted(ranges, key=lambda r: r[2])
    batch, io_plan, used, flushes = [], [], 0, 0

    def flush(reusing):
        nonlocal batch, io_plan, used, flushes
        if not batch:
            return
        flushes += 1
        if direction == "h2d":
            for fo, so, nb in io_plan:
                io.read_into(view[so : so + nb], fo)
        transfer_cache_ranges(direction, [dev], stage, batch, stream, backend="auto")
        if direction == "d2h":
            stream.synchronize()
            for fo, so, nb in io_plan:
                io.write_from(view[so : so + nb], fo)
        elif reusing:
            stream.synchronize()
        batch, io_plan, used = [], [], 0

    for di, do, ho, nb in ordered:
        if used + nb > STAGE:
            flush(True)
        batch.append((di, do, used, nb))
        if (
            io_plan
            and io_plan[-1][0] + io_plan[-1][2] == ho
            and io_plan[-1][1] + io_plan[-1][2] == used
        ):
            io_plan[-1] = (io_plan[-1][0], io_plan[-1][1], io_plan[-1][2] + nb)
        else:
            io_plan.append((ho, used, nb))
        used += nb
    flush(False)
    return flushes


s = torch.cuda.Stream()
n1 = run("d2h", s)
s.synchronize()
expect = dev.clone()
dev.zero_()
torch.cuda.synchronize()
n2 = run("h2d", s)
s.synchronize()

bad = int((dev != expect).sum().item())
print(
    f"{N_PLANES} planes ({N_PLANES*PLANE/1e6:.0f} MB), staging {STAGE/1e6:.0f} MB, "
    f"d2h {n1} 批 / h2d {n2} 批"
)
print(f"不相符的位元組: {bad}  →  {'通過' if bad == 0 else '失敗'}")
os.close(fd)
os.unlink(PATH)
sys.exit(1 if bad else 0)
