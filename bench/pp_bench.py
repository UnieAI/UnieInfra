"""File-L3 prefetch: serial batches (depth 1) vs two in flight (depth 2), O_DIRECT reads."""
import shutil, statistics, sys, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch
from tokenspeed.runtime.cache.l3.file import FileKvStore, FileStoreConfig

PAGES, OBJS, OBJ, BATCH = int(sys.argv[1]), 6, 38 * 1024, 128
path = "/workspace/.l3bench_pp"
shutil.rmtree(path, ignore_errors=True)
cfg = FileStoreConfig(path=path, capacity_gb=20, io_threads=16, o_direct=True, verify_crc=True, fsync=False)
store = FileKvStore(cfg, rank=0, shared_owner_rank=None)
n = PAGES * OBJS
buf = torch.randint(0, 255, (n * OBJ,), dtype=torch.uint8)
keys = [f"k{i}" for i in range(n)]; offs = [i * OBJ for i in range(n)]; sizes = [OBJ] * n
step = BATCH * OBJS
for s in range(0, n, step):  # one put per batch, like backups
    assert all(store.batch_put_from(keys[s:s+step], buf, offs[s:s+step], sizes[s:s+step]))
dst = torch.zeros_like(buf)
batches = [(keys[s:s+step], offs[s:s+step], sizes[s:s+step]) for s in range(0, n, step)]
io = ThreadPoolExecutor(2)

def run(depth):
    t = time.perf_counter(); q = deque(); it = iter(batches)
    def issue():
        b = next(it, None)
        if b is None: return False
        q.append(io.submit(store.batch_get_into, b[0], dst, b[1], b[2])); return True
    while len(q) < depth and issue(): pass
    while q:
        assert all(q.popleft().result())
        while len(q) < depth and issue(): pass
    return (time.perf_counter() - t) * 1e3

res = {1: [], 2: []}
for _ in range(6):
    for d in (1, 2, 2, 1):
        res[d].append(run(d))
assert torch.equal(dst, buf)
mb = n * OBJ / 2**20
for d in (1, 2):
    m = statistics.median(res[d])
    print(f"pages={PAGES} ({mb:.0f} MiB) depth={d}: median {m:.1f} ms  {mb/1024/(m/1e3):.2f} GiB/s")
store.close(); shutil.rmtree(path, ignore_errors=True)
