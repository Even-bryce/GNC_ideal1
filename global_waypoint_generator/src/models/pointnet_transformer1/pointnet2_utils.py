import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

# ==========================================
#  Part 1: 基础几何工具函数 (保持原样，无需动)
# ==========================================

def square_distance(src, dst):
    """
    计算点对之间的欧氏距离平方
    src: [B, N, C]
    dst: [B, M, C]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    根据索引取点
    points: [B, N, C]
    idx: [B, S] or [B, S, K]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    最远点采样 (FPS)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    球半径查询 (Ball Query)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    核心分组函数
    Input:
        xyz: [B, N, 3] (几何坐标)
        points: [B, N, D] (特征)
    Return:
        new_xyz: [B, npoint, 3] (采样后的中心点)
        new_points: [B, npoint, nsample, 3+D] (分组后的特征，包含相对坐标)
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) # [B, npoint, 3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # [B, npoint, nsample]
    
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, 3]
    # 计算相对坐标 (Relative Coordinates)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        # 拼接: [相对坐标(3), 原始特征(D)]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) 
    else:
        new_points = grouped_xyz_norm
        
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    全局分组 (Global Grouping)
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# ==========================================
#  Part 2: Point Transformer 核心模块 (Clean Version)
# ==========================================

class PointTransformerBlock(nn.Module):
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        
        # Q, K, V (保持 Linear 以便于 Reshape 操作)
        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)
        
        # 相对位置编码 MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
        # 投影层 (Projection) - 使用 Conv1d 避免维度报错
        self.linear_p = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )
        
        # FFN (Feed-Forward Network) - 使用 Conv1d
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x, delta):
        """
        x:     [B, dim, N, K]   (特征)
        delta: [B, 3, N, K]     (相对位置)
        """
        B, dim, N, K = x.shape
        
        # === 1. Vector Attention ===
        # 调整为 [B*N, K, dim] 进行矩阵运算
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B*N, K, dim)
        delta_flat = delta.permute(0, 2, 3, 1).contiguous().view(B*N, K, 3)
        
        pos_enc = self.pos_mlp(delta_flat) 
        
        q = self.linear_q(x_flat)
        k = self.linear_k(x_flat)
        v = self.linear_v(x_flat)
        
        relation = q - k + pos_enc
        weight = self.softmax(relation)
        
        # 聚合: [B*N, dim]
        feat = torch.sum(weight * (v + pos_enc), dim=1)
        
        # 还原维度: [B, N, dim] -> [B, dim, N] (适配 Conv1d)
        feat = feat.view(B, N, dim).permute(0, 2, 1)
        
        # === 2. Projection & FFN & Residual ===
        # 投影
        feat = self.linear_p(feat)
        
        # FFN + Residual Connection
        feat = feat + self.ffn(feat)
        
        return F.relu(feat)


class PointTransformerSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, k=16):
        super(PointTransformerSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # MLP 的最后一层作为 Transformer 的维度
        self.trans_dim = mlp[-1] 
        
        # 特征变换层 (Transition Down)
        # 注意: in_channel 必须匹配 (3 + features_dim)
        self.pre_mlp = nn.Sequential(
            nn.Conv2d(in_channel, self.trans_dim, 1),
            nn.BatchNorm2d(self.trans_dim),
            nn.ReLU()
        )
        
        # Transformer Block
        self.transformer = PointTransformerBlock(dim=self.trans_dim, k=k)

    def forward(self, xyz, points):
        """
        Input:
            xyz:    [B, 3, N] (几何坐标，必须是3维)
            points: [B, D, N] (额外特征)
        Output:
            new_xyz:    [B, 3, S]
            new_points: [B, trans_dim, S]
        """
        # 转置为 [B, N, C] 以适配 sample_and_group
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        # 1. 采样与分组
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        # 此时数据形状:
        # new_xyz:    [B, npoint, 3] (中心点坐标)
        # new_points: [B, npoint, nsample, 3+D] (包含相对坐标和特征)
        
        # 2. 准备数据
        # 提取相对坐标 delta (前3维) -> [B, N, K, 3]
        relative_xyz = new_points[:, :, :, :3]
        
        # 提取输入特征 (完整保留 3+D 维，适配 my_model 的 in_channel)
        grouped_features = new_points 

        # 调整维度适配 Conv2d: [B, npoint, nsample, C] -> [B, C, nsample, npoint]
        # 注意：这里 nsample 对应 K，npoint 对应 N
        grouped_features = grouped_features.permute(0, 3, 2, 1)
        
        # 3. 特征变换 (Transition)
        grouped_features = self.pre_mlp(grouped_features) # [B, trans_dim, K, N]
        
        # 调整给 Transformer: [B, trans_dim, N, K]
        grouped_features = grouped_features.permute(0, 1, 3, 2)
        
        # 准备 Delta: [B, 3, N, K]
        delta = relative_xyz.permute(0, 3, 1, 2)

        # 4. Transformer 聚合
        new_points = self.transformer(grouped_features, delta) # [B, trans_dim, N]

        # 5. 输出坐标调整
        # new_xyz [B, N, 3] -> [B, 3, N]
        new_xyz = new_xyz.permute(0, 2, 1)
        
        return new_xyz, new_points


class PointTransformerFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, k=16):
        super(PointTransformerFeaturePropagation, self).__init__()
        self.k = k
        self.out_dim = mlp[-1]
        
        # 特征融合层
        self.linear = nn.Sequential(
            nn.Conv1d(in_channel, self.out_dim, 1),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU()
        )
        
        # Transformer 精修层
        self.transformer = PointTransformerBlock(dim=self.out_dim, k=k)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: [B, 3, N] (Target, Dense)
        xyz2: [B, 3, S] (Source, Sparse)
        points1: [B, D1, N]
        points2: [B, D2, S]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        # 1. 插值 (Interpolation)
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        # 2. 拼接 Skip Connection
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # 3. 融合特征
        new_points = new_points.permute(0, 2, 1) # [B, C_in, N]
        new_points = self.linear(new_points)     # [B, out_dim, N]

        # 4. 局部 Transformer 精修
        # 在 N 个点内部找 k 个邻居
        target_xyz = xyz1 
        
        # k-NN 搜索
        dists = square_distance(target_xyz, target_xyz) 
        _, idx = dists.topk(self.k, dim=-1, largest=False) 
        
        # 收集邻居坐标
        grouped_xyz = index_points(target_xyz, idx) # [B, N, k, 3]
        
        # 收集邻居特征
        feature_for_group = new_points.permute(0, 2, 1) # [B, N, out_dim]
        grouped_feature = index_points(feature_for_group, idx) # [B, N, k, out_dim]
        
        # 调整格式
        grouped_feature = grouped_feature.permute(0, 3, 1, 2) # [B, out_dim, N, k]
        
        # 计算相对坐标 Delta
        delta = grouped_xyz - target_xyz.unsqueeze(2)
        delta = delta.permute(0, 3, 1, 2) # [B, 3, N, k]

        # 运行 Transformer
        refined_points = self.transformer(grouped_feature, delta)

        # 5. 残差连接 (Residual Connection)
        # 防止特征退化
        new_points = new_points + refined_points 

        return F.relu(new_points)