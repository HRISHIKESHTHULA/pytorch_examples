import os

import torch

is_cuda_available = torch.cuda.is_available()

A = torch.randn(2, 2, dtype=torch.complex128)
print(f"{A=}")

if is_cuda_available:
    A.to("cuda")

Eigen_ValA = torch.linalg.eigvals(A)
print(f"{Eigen_ValA}")
