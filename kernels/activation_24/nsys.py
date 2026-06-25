import torch

torch.cuda.init()

x = torch.randn((4096, 4096), device="cuda")
y = torch.randn((4096, 4096), device="cuda")

torch.cuda.synchronize()

for _ in range(20):
    z = x + y

torch.cuda.synchronize()

print("done")