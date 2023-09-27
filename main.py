import torch
from swin_transformer import SwinTransformer

# 创建模型实例
model = SwinTransformer(
    img_size=(224, 224),  # 图像大小
    patch_size=4,         # 图像分块大小
    in_chans=3,           # 输入通道数
    num_classes=1000,     # 类别数（根据您的任务调整）
    embed_dim=96,         # 嵌入维度
    depths=[2, 2, 6, 2],  # 不同阶段的Transformer块深度
    num_heads=[3, 6, 12, 24],  # 不同阶段的注意头数
)

# 加载预训练权重
pretrained_weights = torch.load("path_to_pretrained_weights.pth")
model.load_state_dict(pretrained_weights)
import d2l
