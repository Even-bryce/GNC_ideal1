import torch.nn as nn
import torch
import torch.nn.functional as F
# 引入新的 Transformer 模块
from pointnet2_utils import PointTransformerSetAbstraction, PointTransformerFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        
        # --- SA1: 第一层局部特征提取 ---
        # 替换为 PointTransformerSetAbstraction
        # k=16 是推荐的邻居数，适合稀疏数据
        self.sa1 = PointTransformerSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                                  in_channel=6+additional_channel, mlp=[64, 64, 128], 
                                                  group_all=False, k=16)
        
        # --- SA2: 第二层局部特征提取 ---
        self.sa2 = PointTransformerSetAbstraction(npoint=128, radius=0.4, nsample=64, 
                                                  in_channel=128 + 3, mlp=[128, 128, 256], 
                                                  group_all=False, k=16)
        
        # --- SA3: 全局特征提取 ---
        # 对于全局层 (npoint=None), group_all=True
        # 这里的 Transformer 会在所有点之间做 Attention (如果不改 util 代码逻辑的话)
        # 或者 PointTransformerSetAbstraction 内部处理 group_all 时会退化为全局操作
        self.sa3 = PointTransformerSetAbstraction(npoint=None, radius=None, nsample=None, 
                                                  in_channel=256 + 3, mlp=[256, 512, 1024], 
                                                  group_all=True, k=16)
        
        # --- FP3: 上采样 ---
        # 替换为 PointTransformerFeaturePropagation
        self.fp3 = PointTransformerFeaturePropagation(in_channel=1280, mlp=[256, 256], k=16)
        
        # --- FP2: 上采样 ---
        self.fp2 = PointTransformerFeaturePropagation(in_channel=384, mlp=[256, 128], k=16)
        
        # --- FP1: 最后一层上采样 ---
        # 输入维度: 128 (来自FP2) + 16 (cls_label) + 6 + additional (原始点云特征)
        # 注意: Transformer FP 层会在末尾加一个 Block 精修，能显著改善分割边缘
        self.fp1 = PointTransformerFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128], k=16)
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        # SA 层前向传播
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # 准备 FP1 的 Skip Connection 数据
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        # 拼接: [Label, XYZ, Original Features]
        skip_feat = torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1)
        
        l0_points = self.fp1(l0_xyz, l1_xyz, skip_feat, l1_points)
        
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # 注意: PointNet++ 和 PointTransformer 不需要像 PointNet 那样计算 trans_feat 正则 loss
        # 所以这里的 trans_feat 参数虽然保留了接口，但实际没用到，这是正常的
        total_loss = F.nll_loss(pred, target)

        return total_loss