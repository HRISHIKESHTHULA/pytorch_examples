import torch


is_cuda_available = torch.cuda.is_available()
print(f"is_cuda_available={is_cuda_available}")


a = torch.tensor([
    [1, 2],
    [3, 4]
])
b = torch.tensor([
    [5, 6],
    [7, 8]
])

if is_cuda_available:
    a.to('cuda:0')
    b.to('cuda:0')

c = torch.matmul(a, b)

print(c)
