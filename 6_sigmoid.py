import torch
from torch import nn


is_cuda_available = torch.cuda.is_available()
m = nn.Sigmoid()
