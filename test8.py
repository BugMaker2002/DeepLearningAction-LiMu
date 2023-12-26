import torch
from torch.utils.data import Dataset, ConcatDataset

# 假设有两个数据集类: CustomDataset1 和 CustomDataset2
class CustomDataset1(Dataset):
    def __init__(self):
        # 初始化 dataset1 的数据
        self.data = [torch.randn(3, 32, 32) for _ in range(100)]  # 示例数据，你需要根据实际情况替换成自己的数据
        self.targets = [torch.randint(0, 10, (1,)).item() for _ in range(100)]  # 示例标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class CustomDataset2(Dataset):
    def __init__(self):
        # 初始化 dataset2 的数据
        self.data = [torch.randn(3, 32, 32) for _ in range(50)]  # 示例数据，你需要根据实际情况替换成自己的数据
        self.targets = [torch.randint(0, 1, (1,)).item() for _ in range(50)]  # 示例标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

# 创建自定义数据集实例
dataset1 = CustomDataset1()
dataset2 = CustomDataset2()

# 使用 ConcatDataset 合并两个数据集
concatenated_dataset = ConcatDataset([dataset1, dataset2])

print(concatenated_dataset[0])

