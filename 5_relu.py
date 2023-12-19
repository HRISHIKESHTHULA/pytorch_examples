import os

import torch
from torch import nn


is_cuda_available = torch.cuda.is_available()

m = nn.ReLU()
ip = torch.randn(2)
if is_cuda_available:
    m = m.to("cuda")
    os.environ["ROCBLAS_LAYER"] = "1"
    ip = ip.to("cuda")

print(f"{ip=}")
output = m(ip)
print(f"{output=}")

ip1 = torch.randn(2).unsqueeze(0)
if is_cuda_available:
    ip1 = ip1.to("cuda")
print(f"{ip1=}")
output = torch.cat((m(ip1), m(-ip1)))
print(f"{output=}")

output1 = torch.cat((m(ip), m(-ip)))
print(f"{output1=}")
