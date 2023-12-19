import torch

x = torch.rand(2, 2)
y = torch.rand(2, 2)

xy = torch.matmul(x, y)
print(f"{xy=}")
