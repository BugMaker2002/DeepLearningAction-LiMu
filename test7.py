import torch

# 创建一个需要梯度追踪的张量
x = torch.tensor(2.0, requires_grad=True)

# 进行一些计算
y = x ** 2
z = y * 3

# 分离张量z，得到一个不再需要梯度的普通张量
detached_z = z.detach()

# 对变量z进行反向传播
z.backward()

print(x.grad)  # 输出梯度，因为z有梯度追踪，所以x的梯度为 12.0

# 所以以下代码会报错
print(detached_z.grad)  # 此行代码会输出None，因为detached_z不再具有梯度追踪
