import sys

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
print(f"{empty_t=}")

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
print(f"{full_t=}")

full_t[0][1] = 5
print(f"edit_t={full_t=}")

t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(f"t's size={sys.getsizeof(t.untyped_storage())=}")
float_64_t = t.to(torch.float64)
print(f"{float_64_t=}")
print(f"float64_t's size={sys.getsizeof(float_64_t.untyped_storage())=}")
bfloat_16_t = t.to(torch.bfloat16)
print(f"{bfloat_16_t=}")
print(f"bfloat_16_t's size={sys.getsizeof(bfloat_16_t.untyped_storage())=}")
