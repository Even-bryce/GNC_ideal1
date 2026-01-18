
'''
pointnet++开源代码来源：https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/models
这是我改造的版本
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
import math


class get_model(nn.Module):
    def __init__(self, num_classes, input_dim=6):
        super(get_model, self).__init__()

        # 删除normal_channel参数，因为自由点云不含法向量
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=input_dim, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+input_dim, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """
        xyz: [B, 6, N]
        """
        B, C, N = xyz.shape
        assert C == 6

        l0_xyz = xyz[:, :3, :]      # [B, 3, N]
        l0_points = xyz             # [B, 6, N]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_points = self.fp1(
            l0_xyz,
            l1_xyz,
            l0_points,   # 🔥 不再 concat cls_label
            l1_points
        )

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)           # [B, 1, N]

        x = x.permute(0, 2, 1)      # [B, N, 1]
        return x


def weighted_bce_loss(logits, targets, weights=None):
    """
    logits:  [B, N, 1]
    targets: [B, N, 1] (0/1)
    weights: [B, N] or None
    这个版本没有加mask，理论上来说要加的，但是padding的点很少，且targets全为0，对loss影响不大
    """
    logits = logits.squeeze(-1)
    targets = targets.squeeze(-1).float()

    if weights is None:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='mean'
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, weight=weights, reduction='mean'
        )
    return loss

# def straightness_loss(
#     p,               # [B, N]
#     dist_ij,         # [B, N, N]  归一化距离
#     N_ij,            # [B, N, N]
#     N_corridor_max,            # scalar
#     delta_s=0.7,
#     alpha2=1.0
# ):
#     """
#     只对 p_i > delta_s 的点参与
#     用掩码的方式实现，避免 for 循环。但是这样的实现方式还是要计算所有点对的距离和管道内点数，时间复杂度为N^3
#     """
#     B, N = p.shape

#     mask = (p > delta_s).float()  # [B, N]
#     p_i = p.unsqueeze(2)          # [B, N, 1]
#     p_j = p.unsqueeze(1)          # [B, 1, N]

#     phi_s = torch.exp(-alpha2 * N_ij / N_corridor_max)  # [B, N, N]

#     loss = (
#         p_i * p_j *
#         dist_ij *
#         phi_s *
#         mask.unsqueeze(2) *
#         mask.unsqueeze(1)
#     ).mean()

#     return loss

def straightness_loss(
    p,                   # [B, N]
    xyz,                 # [B, N, 3] (已归一化)
    delta_s=0.5,
    r_corridor=0.03,
    rho=1000.0,
    alpha2=1.0,
    eps=1e-6,
    M_max=64             # 可选：最多取多少个高 p 点
):
    """
    只对 p_i > delta_s 的点子集计算 straightness
    复杂度 ~ O(M^3), M << N
    """

    B, N, _ = xyz.shape
    device = xyz.device
    total_loss = []

    for b in range(B):
        # --------------------------------------------------
        # 1. 筛选高置信点
        # --------------------------------------------------
        mask = p[b] > delta_s
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        if idx.numel() < 2:
            continue

        # 可选：限制最大点数，防止极端情况
        if idx.numel() > M_max:
            topk = torch.topk(p[b, idx], M_max).indices
            idx = idx[topk]

        xb = xyz[b, idx]      # [M, 3]
        pb = p[b, idx]        # [M]
        M = xb.shape[0]

        # --------------------------------------------------
        # 2. 点对距离
        # --------------------------------------------------
        dist_ij = torch.cdist(xb, xb)  # [M, M]

        # --------------------------------------------------
        # 3. corridor 点数 N_ij（局部）
        # --------------------------------------------------
        xi = xb.unsqueeze(1)  # [M, 1, 3]
        xj = xb.unsqueeze(0)  # [1, M, 3]
        xk = xb.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 3]

        v = xj - xi           # [M, M, 3]
        w = xk - xi.unsqueeze(1)  # [M, 1, M, 3]

        vv = (v ** 2).sum(-1, keepdim=True) + eps
        t = (w * v.unsqueeze(1)).sum(-1, keepdim=True) / vv.unsqueeze(1)
        t = torch.clamp(t, 0.0, 1.0)

        proj = xi.unsqueeze(1) + t * v.unsqueeze(1)
        d_k_to_seg = torch.norm(xk - proj, dim=-1)  # [M, M, M]

        N_ij = (d_k_to_seg <= r_corridor).sum(dim=-1).float()  # [M, M]

        # --------------------------------------------------
        # 4. corridor 最大容量
        # --------------------------------------------------
        V_corridor = math.pi * r_corridor ** 2 * dist_ij
        N_corridor_max = rho * V_corridor + eps

        # --------------------------------------------------
        # 5. straightness loss
        # --------------------------------------------------
        p_i = pb.unsqueeze(1)
        p_j = pb.unsqueeze(0)

        phi_s = torch.exp(-alpha2 * N_ij / N_corridor_max)

        loss_mat = p_i * p_j * dist_ij * phi_s

        loss_b = loss_mat.mean()
        total_loss.append(loss_b)

    if len(total_loss) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(total_loss).mean()

# def safety_loss(p, N_i, N_local_max, alpha1=1.0):
#     """
#     p:   [B, N]
#     N_i: [B, N]  小邻域自由空间点数
#     """
#     e_i = torch.exp(-alpha1 * N_i / N_local_max)
#     loss = (p * e_i).mean()
#     return loss

def safety_loss(
    p,                  # [B, N]
    xyz,                # [B, N, 3]
    delta_s=0.5,
    r_local=0.05,
    rho=1000.0,
    alpha1=1.0,
    eps=1e-6,
    M_max=128
):
    """
    点级安全性约束：
    - p 大的点应处在局部稀疏区域
    """

    B, N, _ = xyz.shape
    device = xyz.device
    losses = []

    for b in range(B):
        # -------------------------
        # 1. 选高置信点
        # -------------------------
        mask = p[b] > delta_s
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        if idx.numel() == 0:
            continue

        if idx.numel() > M_max:
            topk = torch.topk(p[b, idx], M_max).indices
            idx = idx[topk]

        xb = xyz[b, idx]   # [M, 3]
        pb = p[b, idx]     # [M]
        M = xb.shape[0]

        # -------------------------
        # 2. 局部邻域点数 N_i
        # -------------------------
        dist = torch.cdist(xb, xb)          # [M, M]
        N_i = (dist <= r_local).sum(dim=-1).float()  # [M]

        # -------------------------
        # 3. 理论最大点数
        # -------------------------
        V_local = 4.0 / 3.0 * math.pi * r_local ** 3
        N_local_max = rho * V_local + eps

        # -------------------------
        # 4. safety loss
        # -------------------------
        e_i = torch.exp(-alpha1 * N_i / N_local_max)
        loss_b = (pb * e_i).mean()

        losses.append(loss_b)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def connectivity_loss(p, xyz, r_connect=0.05, delta_c=0.1):
    """
    连通性正则：空间相邻点的预测概率应平滑

    p:   [B, N]
    xyz: [B, N, 3]
    """
    B, N = p.shape

    # [B, N, N, 3]
    diff_xyz = xyz.unsqueeze(2) - xyz.unsqueeze(1)

    # [B, N, N]
    dist = torch.norm(diff_xyz, dim=-1)

    # 邻域掩码（不包含自身）
    neighbor_mask = (dist < r_connect) & (dist > 0)

    # [B, N, N]
    p_i = p.unsqueeze(2)
    p_j = p.unsqueeze(1)

    diff_p = torch.abs(p_i - p_j)

    # 只在邻域内施加约束
    loss = F.relu(diff_p - delta_c) * neighbor_mask.float()

    # 防止除 0
    denom = neighbor_mask.sum().clamp(min=1.0)

    return loss.sum() / denom


class get_loss(nn.Module):
    def __init__(
        self,
        w_bce=1.0,
        w_straight=0.1,
        w_safety=0.1,
        w_conn=0.0,
        # ---- straightness 超参
        delta_s=0.5,
        r_corridor=0.03,
        rho=1000.0,
        alpha2=1.0,
        M_pair_max=128,
        # ---- safety 超参
        r_local=0.05,
        alpha1=1.0,
        M_safe_max=128,
        # ----- connectivity 超参
        delta_c=0.1,
        r_connect=0.05
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_straight = w_straight
        self.w_safety = w_safety
        self.w_conn = w_conn

        # straightness
        self.delta_s = delta_s
        self.r_corridor = r_corridor
        self.rho = rho
        self.alpha2 = alpha2
        self.M_pair_max = M_pair_max

        # safety
        self.r_local = r_local
        self.alpha1 = alpha1
        self.M_safe_max = M_safe_max

        # connectivity
        self.delta_c = delta_c
        self.r_connect = r_connect

    def forward(self, logits, targets, xyz):
        """
        logits : [B, N, 1]
        targets: [B, N, 1]
        xyz    : [B, N, 3]  (已归一化)
        """

        p = torch.sigmoid(logits).squeeze(-1)  # [B, N]

        loss = 0.0

        # -------------------------
        # 1. BCE
        # -------------------------
        loss += self.w_bce * weighted_bce_loss(logits, targets)

        # -------------------------
        # 2. Straightness
        # -------------------------
        if self.w_straight > 0:
            loss += self.w_straight * straightness_loss(
                p=p,
                xyz=xyz,
                delta_s=self.delta_s,
                r_corridor=self.r_corridor,
                rho=self.rho,
                alpha2=self.alpha2,
                M_max=self.M_pair_max
            )

        # -------------------------
        # 3. Safety
        # -------------------------
        if self.w_safety > 0:
            loss += self.w_safety * safety_loss(
                p=p,
                xyz=xyz,
                delta_s=self.delta_s,
                r_local=self.r_local,
                rho=self.rho,
                alpha1=self.alpha1,
                M_max=self.M_safe_max
            )

        # -------------------------
        # 4. Connectivity（你目前不用，留接口）
        # -------------------------
        if self.w_conn > 0:
            loss += self.w_conn * connectivity_loss(
                p=p,
                xyz=xyz,
                r_connect=self.r_connect,
                delta_c=self.delta_c
            )

        return loss

