import numpy as np

# 创建原始数组
x = np.array([1, 2, 3, 4, 5, 6])

# 使用 reshape 创建新数组
y = x.reshape(2, 3)

# 修改新数组
x[0] = 100

# 查看原始数组
print(x)
print(y)
