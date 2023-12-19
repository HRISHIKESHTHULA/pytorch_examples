import os

import torch
from torch import nn


is_cuda_available = torch.cuda.is_available()

# S 0-1
m = nn.Sigmoid()

ip = torch.randn(5)
if is_cuda_available:
    m = m.to("cuda")
    os.environ["ROCBLAS_LAYER"] = "1"
    ip = ip.to("cuda")

print(f"{ip=}")
output = m(ip)
print(f"{output=}")
