import os

import torch


is_cuda_available = torch.cuda.is_available()
print(f"{is_cuda_available=}")


a = torch.tensor([
    [1., 2.],
    [3., 4.]
])
b = torch.tensor([
    [5., 6.],
    [7., 8.]
])

if is_cuda_available:
    a = a.to('cuda:0')
    b = b.to('cuda:0')

ab = torch.matmul(a, b)

print(f"{ab=}")
