import nvtx
import torch

x = torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize()
for i in range(3):
    nvtx.push_range(message="myrange", color="red")
    y = x @ x
    torch.cuda.synchronize()
    nvtx.pop_range()
print("done")
