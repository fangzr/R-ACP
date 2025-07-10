import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from compressai.ops import LowerBound

class TransformationMatrixGenerator(nn.Module):
    def __init__(self, channel, height, width):
        super(TransformationMatrixGenerator, self).__init__()
        # 添加自适应平均池化层，减少特征尺寸
        self.pool = nn.AdaptiveAvgPool2d((45, 80))  # 将特征从90x160降到45x80
        self.fc1 = nn.Linear(2 * channel * 45 * 80, 256)  # 调整输入大小
        self.fc2 = nn.Linear(256, 6)  # 输出6个参数，表示2D仿射变换矩阵

    def forward(self, feature_0, feature_2):
        # 先将5D张量转换为4D，合并相机和帧维度
        B, N, C, H, W = feature_0.shape  # 假设feature_0的形状为 [B, N, C, H, W]
        
        # Reshape成4D张量以适应自适应池化
        feature_0_reshaped = feature_0.view(B * N, C, H, W)
        feature_2_reshaped = feature_2.view(B * N, C, H, W)

        # 对特征进行池化，减少空间维度
        feature_0_pooled = self.pool(feature_0_reshaped)  # (B*N, channel, 45, 80)
        feature_2_pooled = self.pool(feature_2_reshaped)  # (B*N, channel, 45, 80)

        # Flatten特征
        feature_0_flat = feature_0_pooled.view(B * N, -1)
        feature_2_flat = feature_2_pooled.view(B * N, -1)

        # 拼接第0帧和第2帧的特征
        combined = torch.cat([feature_0_flat, feature_2_flat], dim=1)

        # 通过全连接层生成变换矩阵
        hidden = F.relu(self.fc1(combined))
        transformation_params = self.fc2(hidden)

        # 返回仿射变换的6个参数
        return transformation_params.view(B, N, 6)  # 恢复原来的B,N维度
