import time

import torch


A = torch.rand(1000, 1000, device='cuda')
B = torch.rand(1000, 1000, device='cuda')

t1 = time.time()
C = torch.mm(A, A)
D = torch.mm(B, B)
t2 = time.time()

print(C)
print(D)
print(f"Time taken w/o stream={t2-t1}")

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

torch.cuda.synchronize()

t1 = time.time()
with torch.cuda.stream(s1):
    C = torch.mm(A, A)
with torch.cuda.stream(s2):
    D = torch.mm(B, B)
torch.cuda.synchronize()
t2 = time.time()

print(C)
print(D)
print(f"Time taken with stream={t2-t1}")
