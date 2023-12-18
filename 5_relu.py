import torch
from torch import nn


m = nn.ReLU()
ip = torch.randn(2)
print(f"{ip=}")
output = m(ip)
print(f"{output=}")

ip1 = torch.randn(2).unsqueeze(0)
print(f"{ip1=}")
output = torch.cat((m(ip1), m(-ip1)))
print(f"{output=}")

output1 = torch.cat((m(ip), m(-ip)))
print(f"{output1=}")
