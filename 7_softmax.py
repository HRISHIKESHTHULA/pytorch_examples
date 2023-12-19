import os

from torch.nn.functional import softmax
import torch

is_cuda_available = torch.cuda.is_available()

t = torch.Tensor([1, 2, 3, 4, 5])

if is_cuda_available:
    os.environ["ROCBLAS_LAYER"] = "1"
    t.to("cuda")

softmax_t = softmax(t, dim=0)
print(f"{softmax_t=}")

print(f"{torch.sum(softmax_t)=}")
