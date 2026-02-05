
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
    def __init__(self, num_classes, input_dim=8):
        super(get_model, self).__init__()

        # 删除normal_channel参数，简化设计
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=32, in_channel=input_dim, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+input_dim, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.input_dim = input_dim

    def forward(self, xyz):
        """
        xyz: [B, input_dim, N]
        """
        B, C, N = xyz.shape
        assert C == self.input_dim

        # --- 修改点 1: 拆分坐标和特征 ---
        l0_xyz = xyz[:, :3, :]      # [B, 3, N] -> 物理坐标 (x,y,z)
        l0_features = xyz[:, 3:, :] # [B, 3, N] -> 额外特征 (dist, start, goal)

        # --- 修改点 2: SA1 只传额外特征 ---
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_features)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        # --- 修改点 3: FP1 必须传完整的 6 维数据 ---
        # 因为 fp1 定义的输入通道包含原始 input_dim (128 + 6)
        l0_points = self.fp1(
            l0_xyz,
            l1_xyz,
            xyz,         # <--- 这里保持传完整的 [B, 6, N]
            l1_points
        )

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)           # [B, 1, N]

        x = x.permute(0, 2, 1)      # [B, N, 1]
        return x


# def weighted_bce_loss(logits, targets, weights=None, pos_ratio=10):
#     """
#     logits:  [B, N, 1]
#     targets: [B, N, 1] (0/1)
#     weights: [B, N] (可选，原本的样本权重，如果没有特殊需求可以不传)
#     pos_ratio: 正样本权重系数。建议设为 (总点数/正样本数)
#     """
#     # 1. 维度调整
#     logits = logits.squeeze(-1)   # [B, N]
#     targets = targets.squeeze(-1).float() # [B, N]

#     # 2. 核心修改：定义 pos_weight
#     # 它的作用是：预测错了正样本(1)，惩罚是预测错负样本(0)的 pos_ratio 倍
#     # 注意：必须把它放到和 logits 同一个 device (cuda) 上
#     pos_weight = torch.tensor([pos_ratio], device=logits.device)

#     # 3. 计算 Loss
#     # 注意这里多传了一个 pos_weight 参数
#     if weights is None:
#         loss = F.binary_cross_entropy_with_logits(
#             logits, 
#             targets, 
#             pos_weight=pos_weight,  # <--- 关键修改
#             reduction='mean'
#         )
#     else:
#         # 如果你还想保留原来的 weights (比如某种空间掩码)，可以同时用
#         loss = F.binary_cross_entropy_with_logits(
#             logits, 
#             targets, 
#             weight=weights, 
#             pos_weight=pos_weight,  # <--- 关键修改
#             reduction='mean'
#         )
        
#     return loss

def focal_loss(logits, targets, weights=None, alpha=0.9, gamma=2.0):
    """
    【严格遵循 Soft Focal Loss 公式版本】
    
    Formula:
        Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Where:
        p_t = y * p + (1 - y) * (1 - p)
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
    
    Args:
        logits:  [B, N] (未经过 Sigmoid 的原始输出)
        targets: [B, N] (0~1 之间的软标签)
        weights: [B, N] (可选空间权重)
        alpha:   必须在 [0, 1] 之间。平衡正负样本权重。
        gamma:   聚焦系数，推荐 2.0。
    """
    # 1. 维度对齐
    logits = logits.squeeze(-1) if logits.dim() > 2 else logits # [B, N]
    targets = targets.squeeze(-1).float() if targets.dim() > 2 else targets # [B, N]

    # 2. 计算概率 p (Sigmoid)
    probs = torch.sigmoid(logits)

    # 3. 计算 p_t (根据公式：预测与标签的一致性概率)
    # 这里的 targets 就是公式里的 y，probs 就是 p
    # p_t = y*p + (1-y)*(1-p)
    p_t = targets * probs + (1 - targets) * (1 - probs)

    # 4. 计算 Modulating Factor (聚焦因子)
    # 公式：(1 - p_t)^gamma
    modulating_factor = (1 - p_t) ** gamma

    # 5. 计算 Alpha_t (平衡因子)
    # 公式：alpha_t = alpha * y + (1 - alpha) * (1 - y)
    # 注意：严格按照公式，alpha 必须在 [0, 1] 之间
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    else:
        alpha_t = 1.0

    # 6. 计算基础 BCE Loss
    # F.binary_cross_entropy_with_logits 对应公式中的 -[y log p + (1-y) log(1-p)]
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # 7. 组合最终 Loss
    # Loss = alpha_t * (1 - p_t)^gamma * BCE
    loss = alpha_t * modulating_factor * bce_loss

    # 8. 处理额外的空间权重 (weights)
    if weights is not None:
        weights = weights.squeeze(-1) if weights.dim() > loss.dim() else weights
        loss = loss * weights

    return loss.mean()

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
    p,                  # [B, N]
    xyz,                # [B, N, 3] (已归一化)
    delta_s=0.5,        # 筛选高分点的阈值
    delta_d=0.2,        # <--- 新增参数：只有距离大于该值的点对才计算 Loss
    r_corridor=0.03,
    rho=20000.0,
    alpha2=1.0,
    eps=1e-6,
    M_max=128           # 限制计算复杂度的最大点数
):
    """
    修改版 straightness_loss:
    1. 仅当 dist_ij > delta_d 时计算 Loss
    2. 去除了公式中的 * dist_ij，不再显式奖励长距离
    """
    
    B, N, _ = xyz.shape
    device = xyz.device
    total_loss = []

    for b in range(B):
        # --------------------------------------------------
        # 1. 筛选高置信点 (High Confidence Points)
        # --------------------------------------------------
        mask = p[b] > delta_s
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        # 如果高分点太少，无法构成任何点对，跳过
        if idx.numel() < 2:
            continue

        # 限制计算量：如果有太多高分点，只取分数最高的 M_max 个
        if idx.numel() > M_max:
            topk = torch.topk(p[b, idx], M_max).indices
            idx = idx[topk]

        xb = xyz[b, idx]      # [M, 3]
        pb = p[b, idx]        # [M]
        M = xb.shape[0]

        # --------------------------------------------------
        # 2. 计算距离并生成距离掩码 (Distance Mask)
        # --------------------------------------------------
        dist_ij = torch.cdist(xb, xb)  # [M, M]
        
        # <--- 修改点 A: 生成 Mask，只有距离足够远的点对才有效
        valid_pair_mask = (dist_ij > delta_d).float()
        
        # 如果当前没有满足距离要求的点对，跳过
        num_valid_pairs = valid_pair_mask.sum()
        if num_valid_pairs < 1:
            continue

        # --------------------------------------------------
        # 3. 计算 Corridor 内的阻挡点数 N_ij
        #    (计算逻辑不变，依然基于几何投影)
        # --------------------------------------------------
        xi = xb.unsqueeze(1)          # [M, 1, 3]
        xj = xb.unsqueeze(0)          # [1, M, 3]
        xk = xb.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 3]

        v = xj - xi                   # [M, M, 3] 向量 i->j
        
        # 计算投影比例 t
        # 注意：为了防止除以0，分母加 eps。
        # 其实 dist_ij 已经在上面算过了，vv = dist_ij^2，可以复用优化，但为了逻辑清晰保留原样
        vv = (v ** 2).sum(-1, keepdim=True) + eps 
        
        w = xk - xi.unsqueeze(1)      # [M, 1, M, 3] 向量 i->k
        
        t = (w * v.unsqueeze(1)).sum(-1, keepdim=True) / vv.unsqueeze(1)
        t = torch.clamp(t, 0.0, 1.0)  # 限制在线段内

        proj = xi.unsqueeze(1) + t * v.unsqueeze(1) # 点 k 在线段 ij 上的投影点
        d_k_to_seg = torch.norm(xk - proj, dim=-1)  # [M, M, M] 点 k 到线段 ij 的垂直距离

        # 统计落在圆柱体内的点数
        N_ij = (d_k_to_seg <= r_corridor).sum(dim=-1).float()  # [M, M]

        # --------------------------------------------------
        # 4. 计算理论最大点数 (用于归一化)
        # --------------------------------------------------
        V_corridor = math.pi * r_corridor ** 2 * dist_ij
        N_corridor_max = rho * V_corridor + eps

        # --------------------------------------------------
        # 5. 计算 Loss
        # --------------------------------------------------
        p_i = pb.unsqueeze(1)
        p_j = pb.unsqueeze(0)

        # 直线性惩罚项：中间阻挡点 N_ij 越少，phi_s 越大(接近1)，Loss 越小(越负)
        phi_s = torch.exp(-alpha2 * N_ij / N_corridor_max)

        # <--- 修改点 B: 计算 Loss 矩阵
        # 原公式: -1 * p_i * p_j * dist_ij * phi_s
        # 新公式: -1 * p_i * p_j * phi_s  (去掉了 dist_ij)
        loss_mat = -1 * p_i * p_j * phi_s

        # <--- 修改点 C: 应用距离 Mask 并计算平均值
        # 只对 valid_pair_mask 为 1 的位置求和，并除以有效对的数量
        # 这样避免了大量无效的 0 值拉低了 Loss 的绝对值
        loss_b = (loss_mat * valid_pair_mask).sum() / (num_valid_pairs + eps)
        
        total_loss.append(loss_b)

    # 如果所有 batch 都没有有效点对，返回 0
    if len(total_loss) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

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
    rho=20000.0,
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
        w_straight=3,
        w_safety=0.2,
        w_conn=0.0,
        # ---- focal loss 超参
        alpha=0.95,
        gamma=2.0, 
        # ---- straightness 超参
        delta_s=0.5,
        delta_d=0.2,
        r_corridor=0.03,
        rho=20000.0,
        alpha2=1.0,
        M_pair_max=128,
        # ---- safety 超参
        r_local=0.05,
        alpha1=1.0,
        M_safe_max=256,
        # ----- connectivity 超参
        delta_c=0.1,
        r_connect=0.05
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_straight = w_straight
        self.w_safety = w_safety
        self.w_conn = w_conn
        self.alpha = alpha
        self.gamma = gamma 

        # straightness
        self.delta_s = delta_s
        self.delta_d = delta_d
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
        loss += self.w_bce * focal_loss(logits, targets, alpha=self.alpha, gamma=self.gamma)

        # -------------------------
        # 2. Straightness
        # -------------------------
        if self.w_straight > 0:
            loss += self.w_straight * straightness_loss(
                p=p,
                xyz=xyz,
                delta_s=self.delta_s,
                delta_d=self.delta_d,
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

