import torch

x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = torch.matmul(x, y)
print(z)
