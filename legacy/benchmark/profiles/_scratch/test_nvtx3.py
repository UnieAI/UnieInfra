import torch

x = torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize()
for i in range(3):
    torch.cuda.nvtx.range_push("myrange")
    y = x @ x
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
