import torch
from torch import Tensor


"""
| 1 2 3 |
| 4 5 6 |
"""
t1 = Tensor([[1, 2, 3], [4, 5, 6]])
t2 = Tensor([[-1, -2, -3], [-4, -5, -6]])
print(f"{t1=}")
print(f"{t2=}")
t1_plus_t2 = t1 + t2

print(f"{t1_plus_t2=}")

empty_t = torch.empty((5,))

print(f"{empty_t}")

ones_t = torch.ones(5)
print(f"{ones_t=}")

zeros_t = torch.zeros(5)
print(f"{zeros_t=}")

ones_tm1 = torch.ones(5, 5)
print(f"{ones_tm1=}")

ones_tm2 = torch.ones(5, 5)
print(f"{ones_tm2=}")

ones_tm3 = ones_tm1 + ones_tm2
print(f"{ones_tm3=}")

full_t = torch.full((2, 2), 3)
print(full_t)
