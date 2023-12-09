import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

pred = torch.rand(size=(32, 21, 320, 480), dtype=torch.float32)
truth = torch.randint(0, 21, size=(32, 320, 480))

result = loss(pred, truth)
print(result.shape)
print(result)

