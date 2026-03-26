import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入底层 CUDA 算子
from src.models.pointnet_transfomer2.lib.pointops.functions import pointops

# ==========================================
#  Part 1: 官方 Point Transformer 核心组件
# ==========================================

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x

class PointTransformerBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        share_planes = 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        # 如果 self.c > 3，说明除了坐标还有额外特征，将其拼接
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x

# ==========================================
#  Part 2: 桥接 Wrapper (无缝替换旧模型)
# ==========================================

class get_model(nn.Module):
    def __init__(self, num_classes, input_dim=8):
        super(get_model, self).__init__()
        self.input_dim = input_dim
        
        # 实例化官方 Backbone，使用标准的块数分布
        # c = 你的总输入维度 (比如 8)。k = num_classes (你的任务应该是 1)
        self.backbone = PointTransformerSeg(
            block=PointTransformerBlock, 
            blocks=[2, 3, 4, 6, 3], 
            c=input_dim, 
            k=num_classes
        )

    def forward(self, xyz):
        """
        完美兼容你原来的输入输出格式:
        输入 xyz: [B, input_dim, N] 
        输出 out: [B, N, num_classes] (通常 num_classes=1)
        """
        B, C_in, N = xyz.shape
        device = xyz.device
        
        # 1. 转换维度: [B, C, N] -> [B, N, C]
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        
       # 2. 高效压平为 pxo 格式
        # 加上 .contiguous() 强制在显存中开辟一块紧凑的连续内存
        p = xyz_trans[..., :3].contiguous().view(B * N, 3)
        
        # x: 所有点的额外特征，形状 [B*N, input_dim-3]
        if self.input_dim > 3:
            x = xyz_trans[..., 3:].contiguous().view(B * N, self.input_dim - 3)
        else:
            x = None
            
        # o: Offset 张量，极速生成法 (并确保其连续)
        o = (torch.arange(1, B + 1, dtype=torch.int32, device=device) * N).contiguous()
        
        # 3. 送入开源 SOTA 模型进行前向计算
        out = self.backbone([p, x, o])  # 输出维度: [B*N, num_classes]
        
        # 4. 重新拉伸回你的预测框架所需的维度
        out = out.view(B, N, -1)  # 输出维度: [B, N, num_classes]
        
        return out


# ==========================================
#  Part 3: 你的专属 Loss 模块 (完全保持原样)
# ==========================================

def focal_loss(logits, targets, weights=None, alpha=0.9, gamma=2.0):
    logits = logits.squeeze(-1) if logits.dim() > 2 else logits
    targets = targets.squeeze(-1).float() if targets.dim() > 2 else targets
    probs = torch.sigmoid(logits)
    p_t = targets * probs + (1 - targets) * (1 - probs)
    modulating_factor = (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    else:
        alpha_t = 1.0
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = alpha_t * modulating_factor * bce_loss
    if weights is not None:
        weights = weights.squeeze(-1) if weights.dim() > loss.dim() else weights
        loss = loss * weights
    return loss.mean()

def straightness_loss(p, xyz, delta_s=0.5, delta_d=0.2, r_corridor=0.03, rho=20000.0, alpha2=1.0, eps=1e-6, M_max=128):
    B, N, _ = xyz.shape
    device = xyz.device
    total_loss = []
    for b in range(B):
        mask = p[b] > delta_s
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() < 2: continue
        if idx.numel() > M_max:
            topk = torch.topk(p[b, idx], M_max).indices
            idx = idx[topk]
        xb = xyz[b, idx]
        pb = p[b, idx]
        M = xb.shape[0]
        dist_ij = torch.cdist(xb, xb)
        valid_pair_mask = (dist_ij > delta_d).float()
        num_valid_pairs = valid_pair_mask.sum()
        if num_valid_pairs < 1: continue
        xi, xj = xb.unsqueeze(1), xb.unsqueeze(0)
        xk = xb.unsqueeze(0).unsqueeze(0)
        v = xj - xi
        vv = (v ** 2).sum(-1, keepdim=True) + eps
        w = xk - xi.unsqueeze(1)
        t = (w * v.unsqueeze(1)).sum(-1, keepdim=True) / vv.unsqueeze(1)
        t = torch.clamp(t, 0.0, 1.0)
        proj = xi.unsqueeze(1) + t * v.unsqueeze(1)
        d_k_to_seg = torch.norm(xk - proj, dim=-1)
        N_ij = (d_k_to_seg <= r_corridor).sum(dim=-1).float()
        V_corridor = math.pi * r_corridor ** 2 * dist_ij
        N_corridor_max = rho * V_corridor + eps
        p_i, p_j = pb.unsqueeze(1), pb.unsqueeze(0)
        phi_s = torch.exp(-alpha2 * N_ij / N_corridor_max)
        loss_mat = -1 * p_i * p_j * phi_s
        loss_b = (loss_mat * valid_pair_mask).sum() / (num_valid_pairs + eps)
        total_loss.append(loss_b)
    if len(total_loss) == 0: return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(total_loss).mean()

def safety_loss(p, xyz, delta_s=0.5, r_local=0.05, rho=20000.0, alpha1=1.0, eps=1e-6, M_max=128):
    B, N, _ = xyz.shape
    device = xyz.device
    losses = []
    for b in range(B):
        mask = p[b] > delta_s
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0: continue
        if idx.numel() > M_max:
            topk = torch.topk(p[b, idx], M_max).indices
            idx = idx[topk]
        xb = xyz[b, idx]
        pb = p[b, idx]
        dist = torch.cdist(xb, xb)
        N_i = (dist <= r_local).sum(dim=-1).float()
        V_local = 4.0 / 3.0 * math.pi * r_local ** 3
        N_local_max = rho * V_local + eps
        e_i = torch.exp(-alpha1 * N_i / N_local_max)
        losses.append((pb * e_i).mean())
    if len(losses) == 0: return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()

def connectivity_loss(p, xyz, r_connect=0.05, delta_c=0.1):
    diff_xyz = xyz.unsqueeze(2) - xyz.unsqueeze(1)
    dist = torch.norm(diff_xyz, dim=-1)
    neighbor_mask = (dist < r_connect) & (dist > 0)
    p_i, p_j = p.unsqueeze(2), p.unsqueeze(1)
    diff_p = torch.abs(p_i - p_j)
    loss = F.relu(diff_p - delta_c) * neighbor_mask.float()
    denom = neighbor_mask.sum().clamp(min=1.0)
    return loss.sum() / denom

class get_loss(nn.Module):
    def __init__(self, w_bce=1.0, w_straight=3, w_safety=0.2, w_conn=0.0, alpha=0.95, gamma=2.0, delta_s=0.5, delta_d=0.2, r_corridor=0.03, rho=20000.0, alpha2=1.0, M_pair_max=128, r_local=0.05, alpha1=1.0, M_safe_max=256, delta_c=0.1, r_connect=0.05):
        super().__init__()
        self.w_bce = w_bce; self.w_straight = w_straight; self.w_safety = w_safety; self.w_conn = w_conn
        self.alpha = alpha; self.gamma = gamma
        self.delta_s = delta_s; self.delta_d = delta_d; self.r_corridor = r_corridor; self.rho = rho; self.alpha2 = alpha2; self.M_pair_max = M_pair_max
        self.r_local = r_local; self.alpha1 = alpha1; self.M_safe_max = M_safe_max
        self.delta_c = delta_c; self.r_connect = r_connect

    def forward(self, logits, targets, xyz, full_points=None):
        if isinstance(logits, (tuple, list)): logits = logits[0]
        if xyz.shape[1] == 8 or xyz.shape[1] == 3: xyz_phys = xyz.permute(0, 2, 1)
        else: xyz_phys = xyz
        xyz_phys = xyz_phys[..., :3]
        p = torch.sigmoid(logits)
        if p.dim() == 3: p = p.squeeze(-1)
        
        # --- 【核心拦截器】利用负索引生成 Ignore Mask ---
        bce_weights = None
        if full_points is not None and full_points.shape[1] >= 6:

            f_start = full_points[:, 3, :]
            f_goal  = full_points[:, 4, :]
            
            # 找出起终点 (特征值趋近于 1.0)
            is_start_end = ((f_start > 0.95) | (f_goal > 0.95)).float()
            
            # 权重反转：起终点的权重被无情置为 0，其它普通点权重保留为 1
            bce_weights = 1.0 - is_start_end

        loss = 0.0
        # 将拦截器权重传入 focal_loss，起终点不再产生分类梯度！
        loss += self.w_bce * focal_loss(logits, targets, weights=bce_weights, alpha=self.alpha, gamma=self.gamma)
        
        # 几何 Loss 依然保留起点和终点，因为它们是必须连通的物理锚点
        if self.w_straight > 0:
            loss += self.w_straight * straightness_loss(p=p, xyz=xyz_phys, delta_s=self.delta_s, delta_d=self.delta_d, r_corridor=self.r_corridor, rho=self.rho, alpha2=self.alpha2, M_max=self.M_pair_max)
        if self.w_safety > 0:
            loss += self.w_safety * safety_loss(p=p, xyz=xyz_phys, delta_s=self.delta_s, r_local=self.r_local, rho=self.rho, alpha1=self.alpha1, M_max=self.M_safe_max)
        if self.w_conn > 0:
            loss += self.w_conn * connectivity_loss(p=p, xyz=xyz_phys, r_connect=self.r_connect, delta_c=self.delta_c)
        return loss