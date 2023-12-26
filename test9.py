import torch
from torch import nn
net = nn.Linear(30, 10)
x = torch.randn(size=(2, 8, 30), dtype=torch.float32)
print(net(x).shape)
