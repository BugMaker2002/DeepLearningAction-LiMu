import torch
from d2l import torch as d2l

boxes1 = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [2.0, 3.0, 3.6, 4.6]])
boxes2 = torch.tensor([[3.5, 4.5, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0]])

box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                          (boxes[:, 3] - boxes[:, 1]))
areas1 = box_area(boxes1)
areas2 = box_area(boxes2)

print('areas1:\n', areas1)
print('areas2:\n', areas2)

'''
在广播机制中，张量在进行广播时会从尾部开始比较维度。对于给定的两个张量，广播操作会从最右边（即最后一个维度）开始匹配，并尝试扩展维度以使它们兼容。
在两个张量进行逐元素操作时，从最右边（尾部）开始比较各个维度。对于形状为 2x1x2 的张量和形状为 2x2 的张量：
①第一个张量的最右边维度是 2，而第二个张量的最右边维度也是 2，它们的尺寸相等，因此在这个维度上是兼容的。
然而，当两个张量的维度不匹配时，广播机制会尝试在较小维度的张量中插入尺寸为 1 的维度，使得两个张量的维度相等。
②由于第一个张量有一个尺寸为 1 的中间维度（1x2），而第二个张量的维度是 2x2，广播机制会在第一个张量的第二个维度（索引为1的维度）进行扩展，
使其尺寸从 1 扩展为 2，以便与第二个张量的维度匹配。
③之后，再在2x2向量的第一个维度上扩充，变成2x2x2
在这里，两个向量就可以进行比较了
'''

inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
# print(boxes1[:, None, :2])
# print(boxes2[:, :2])
# print(inter_upperlefts)  # 维度为2x2x2，boxes1的每一行与boxes2的每一行相比后的结果

inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
# print(inter_lowerrights)

'''
.clamp(min=0) 是一个用于张量操作的函数，它将张量中的值限制在一个指定范围内。
具体来说，.clamp(min=0) 的作用是将张量中的元素限制在不小于指定最小值（min=0）的范围内。如果张量中的元素小于指定的最小值，则将其设为指定的最小值。
'''
inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
# print(inter_lowerrights - inter_upperlefts)
# print(inters)
inter_areas = inters[:, :, 0] * inters[:, :, 1]
print(inter_areas.shape)
print(inter_areas)
union_areas = areas1[:, None] + areas2 - inter_areas
# 在这个式子的运算过程当中，由于广播机制，area1被扩展成了：[[4,4],[2.56,2.56]]，area2被扩展成了：[[0.25,4],[0.25,4]]
# inter_areas表示的结果的意义就是：inter_areas的第一行是area1的第一行与area2的所有行的IOU值，inter_areas第一行的长度即area2的总行数，以此类推
# print(inter_areas / union_areas)
