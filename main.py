import torch
import torch.nn as nn
x = torch.tensor([2.0, 3.0])
x.requires_grad_(True)
with torch.no_grad():
    z = x * 2
z.sum().backward()
print(x.grad)
