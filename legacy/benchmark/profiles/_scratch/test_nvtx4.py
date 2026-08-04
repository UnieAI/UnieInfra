import torch

x = torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize()
for i in range(6):
    torch.cuda.nvtx.range_push(f"myrange")
    y = x @ x
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
