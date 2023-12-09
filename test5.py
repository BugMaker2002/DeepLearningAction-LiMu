import torch
import torch.nn as nn
import torch.nn.functional as F
loss = nn.CrossEntropyLoss()
# labels = torch.arange(0, 128)
# labels = torch.cat([labels, labels], axis=0)
# logits = torch.rand(size=(256, 256))
# print(loss(logits, labels))
target = torch.arange(4)
output = torch.rand(4, 10)
print(output)
_, pred = output.topk(5, 1, True, True)
print(pred)
pred = pred.t()
print(pred)
print(target.view(1, -1).expand_as(pred))
correct = pred.eq(target.view(1, -1).expand_as(pred))
print(correct)
correct_k = correct[:5].flatten().float().sum(0, keepdim=True)
print(correct_k)
print(correct_k.mul_(100.0 / 256))




