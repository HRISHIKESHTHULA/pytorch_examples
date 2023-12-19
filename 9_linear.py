import torch.nn as nn
import torch


m = nn.Linear(8, 12)
ip = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8])
op = m(ip)
print(f"{op=}")
print(f"{list(m.parameters())=}")
