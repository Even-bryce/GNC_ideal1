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
    def __init__(self, in_planes, out_planes, share_planes=4, nsample=16):
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

# class TransitionUp(nn.Module):
#     def __init__(self, in_planes, out_planes=None):
#         super().__init__()
#         if out_planes is None:
#             self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
#             self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
#         else:
#             self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
#             self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
#     def forward(self, pxo1, pxo2=None):
#         if pxo2 is None:
#             _, x, o = pxo1  # (n, 3), (n, c), (b)
#             x_tmp = []
#             for i in range(o.shape[0]):
#                 if i == 0:
#                     s_i, e_i, cnt = 0, o[0], o[0]
#                 else:
#                     s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
#                 x_b = x[s_i:e_i, :]
#                 x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
#                 x_tmp.append(x_b)
#             x = torch.cat(x_tmp, 0)
#             x = self.linear1(x)
#         else:
#             p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
#             x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
#         return x

# class TransitionUp(nn.Module):

#     def __init__(self, in_planes, out_planes=None, nsample=8):
#         super().__init__()
#         # 这里的 nsample=3 对应原版插值找 3 个最近邻，你也可以调成 8 来获得更大的感受野
#         self.nsample = nsample

#         if out_planes is None:
#             self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
#             self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
#         else:
#             self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
#             self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

#             # ==========================================
#             # 💡 彻底重构：基于语义和位置的 Local Cross Attention Upsampling
#             # ==========================================
#             # 1. 特征映射：Q (来自解码器底层特征 x1), K/V (来自编码器稀疏高层特征 x2)
#             self.linear_q = nn.Linear(out_planes, out_planes)
#             self.linear_k = nn.Linear(out_planes, out_planes)
#             self.linear_v = nn.Linear(out_planes, out_planes)

#             # 2. 相对位置编码 (把三维物理距离转化为特征向量)
#             self.linear_p = nn.Sequential(
#                 nn.Linear(3, 3),
#                 nn.BatchNorm1d(3),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(3, out_planes)
#             )
#             # 3. 权重生成 MLP (用来融合 语义差异 Q-K 和 物理距离 PE)
#             self.linear_w = nn.Sequential(
#                 nn.BatchNorm1d(out_planes),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(out_planes, out_planes // 4),
#                 nn.BatchNorm1d(out_planes // 4),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(out_planes // 4, out_planes)
#             )
#             self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, pxo1, pxo2=None):
#         if pxo2 is None:
#             # 最深层的处理逻辑 (保留原样)
#             _, x, o = pxo1
#             x_tmp = []
#             for i in range(o.shape[0]):
#                 if i == 0:
#                     s_i, e_i, cnt = 0, o[0], o[0]
#                 else:
#                     s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
#                 x_b = x[s_i:e_i, :]
#                 x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
#                 x_tmp.append(x_b)
#             x = torch.cat(x_tmp, 0)
#             x = self.linear1(x)
#         else:
#             # pxo1: Encoder 特征 (密集, 包含丰富局部细节, N1个点)
#             # pxo2: Decoder 特征 (稀疏, 包含宏观语义骨架, N2个点)
#             p1, x1, o1 = pxo1
#             p2, x2, o2 = pxo2
#             x1_proj = self.linear1(x1)  # (N1, C)
#             x2_proj = self.linear2(x2)  # (N2, C)
#             if hasattr(self, 'linear_q'):
#                 # ==========================================
#                 # 💡 核心过程：用 Query&Group 替代物理插值
#                 # ==========================================
#                 # 1. 找邻居：拿着密集点(p1)去稀疏点(p2)里找 nsample 个最近邻
#                 # group_features 维度: (N1, nsample, 3 + C) -> 前3个通道是相对物理坐标
#                 group_features = pointops.queryandgroup(self.nsample, p2, p1, x2_proj, None, o2, o1, use_xyz=True)

#                 # 分离出 相对物理坐标 (p_r) 和 邻居的语义特征 (x_k)
#                 p_r = group_features[:, :, 0:3]  # (N1, nsample, 3)
#                 x_k_sparse = group_features[:, :, 3:]   # (N1, nsample, C)

#                 # 2. 生成 Query, Key, Value
#                 q = self.linear_q(x1_proj).unsqueeze(1) # (N1, 1, C)
#                 k = self.linear_k(x_k_sparse)           # (N1, nsample, C)
#                 v = self.linear_v(x_k_sparse)           # (N1, nsample, C)

#                 # 3. 处理相对位置编码
#                 # 把 (N1, nsample, 3) 压平过 MLP 再变回来
#                 pe = p_r.view(-1, 3)
#                 for i, layer in enumerate(self.linear_p):
#                     pe = layer(pe)
#                 pe = pe.view(p_r.shape[0], p_r.shape[1], -1) # (N1, nsample, C)
               
#                 # 4. 计算融合了 语义(q-k) 和 空间距离(pe) 的 Attention 权重
#                 w = q - k + pe  # (N1, nsample, C)
               
#                 # 过权重生成网络
#                 w_flat = w.view(-1, w.shape[-1])
#                 for i, layer in enumerate(self.linear_w):
#                     w_flat = layer(w_flat)

#                 w = w_flat.view(w.shape[0], w.shape[1], -1)
#                 # 在 nsample (即候选邻居) 维度上做 Softmax 归一化
#                 attn_weight = self.softmax(w) # (N1, nsample, C)

#                 # 5. 特征聚合：按权重吸收邻居的特征 (Value) 和 位置编码 (PE)
#                 x2_interp = (attn_weight * (v + pe)).sum(dim=1) # (N1, C)
               
#                 # 6. Skip Connection 融合
#                 x = x1_proj + x2_interp
             
#             else:
#                 # 兼容原始插值
#                 x2_interp = pointops.interpolation(p2, p1, x2_proj, o2, o1)
#                 x = x1_proj + x2_interp
#         return x
    

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None, nsample=4):
        super().__init__()
        self.nsample = nsample 
        
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            
            # ==========================================
            # 💡 机制一：上采样 Local Cross Attention (1-to-N, 找邻居)
            # ==========================================
            self.up_q = nn.Linear(out_planes, out_planes)
            self.up_k = nn.Linear(out_planes, out_planes)
            self.up_v = nn.Linear(out_planes, out_planes)
            
            self.up_p = nn.Sequential(
                nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes)
            )
            self.up_w = nn.Sequential(
                nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True),
                nn.Linear(out_planes, out_planes // 4), nn.BatchNorm1d(out_planes // 4), nn.ReLU(inplace=True),
                nn.Linear(out_planes // 4, out_planes)
            )
            self.up_softmax = nn.Softmax(dim=1)
            
            # ==========================================
            # 💡 机制二：Skip Connection Cross Attention (1-to-1, 门控过滤)
            # ==========================================
            # 专门为跳跃连接准备的独立映射层
            self.skip_q = nn.Linear(out_planes, out_planes)
            self.skip_k = nn.Linear(out_planes, out_planes)
            self.skip_v = nn.Linear(out_planes, out_planes)
            
            # 门控权重生成网络 (最后使用 Sigmoid 将特征压到 0~1 之间)
            self.skip_w = nn.Sequential(
                nn.Linear(out_planes, out_planes // 4),
                nn.BatchNorm1d(out_planes // 4),
                nn.ReLU(inplace=True),
                nn.Linear(out_planes // 4, out_planes),
                nn.Sigmoid() 
            )
            # ==========================================
            
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            # 最深层的处理逻辑 (保留原样)
            _, x, o = pxo1
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
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            
            x1_proj = self.linear1(x1)  # (N1, C)
            x2_proj = self.linear2(x2)  # (N2, C)
            
            if hasattr(self, 'up_q'):
                # --------------------------------------------------
                # 阶段一：基于语义和距离的上采样 (Upsampling)
                # --------------------------------------------------
                group_features = pointops.queryandgroup(self.nsample, p2, p1, x2_proj, None, o2, o1, use_xyz=True)
                
                p_r = group_features[:, :, 0:3]         # (N1, nsample, 3)
                x_k_sparse = group_features[:, :, 3:]   # (N1, nsample, C)
                
                q_up = self.up_q(x1_proj).unsqueeze(1)  # (N1, 1, C)
                k_up = self.up_k(x_k_sparse)            # (N1, nsample, C)
                v_up = self.up_v(x_k_sparse)            # (N1, nsample, C)
                
                pe = p_r.view(-1, 3)
                for layer in self.up_p: pe = layer(pe)
                pe = pe.view(p_r.shape[0], p_r.shape[1], -1)
                
                w_up = q_up - k_up + pe
                w_up_flat = w_up.view(-1, w_up.shape[-1])
                for layer in self.up_w: w_up_flat = layer(w_up_flat)
                w_up = w_up_flat.view(w_up.shape[0], w_up.shape[1], -1)
                
                attn_weight_up = self.up_softmax(w_up) 
                
                # 获得上采样后的高层语义特征
                x2_interp = (attn_weight_up * (v_up + pe)).sum(dim=1) # (N1, C)
                
                # --------------------------------------------------
                # 阶段二：Skip Connection 的 1对1 Cross Attention
                # --------------------------------------------------
                # Query: 刚刚融合好的高层宏观语义 (来自阶段一的输出)
                q_skip = self.skip_q(x2_interp)
                
                # Key/Value: 当前点在编码器中提取的底层微观几何特征
                k_skip = self.skip_k(x1_proj)
                v_skip = self.skip_v(x1_proj)
                
                # 计算门控权重 (Q - K 经过 MLP 和 Sigmoid)
                attn_weight_skip = self.skip_w(q_skip - k_skip) # (N1, C) 范围在 (0, 1) 之间
                
                # 最终融合：宏观语义作为主体，吸收被门控过滤后的底层微观特征
                x = x2_interp + attn_weight_skip * v_skip
                
            else:
                x2_interp = pointops.interpolation(p2, p1, x2_proj, o2, o1)
                x = x1_proj + x2_interp
                
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

# class PointTransformerSeg(nn.Module):
#     def __init__(self, block, blocks, c=6, k=13, dropout_p=0):
#         super().__init__()
#         self.c = c
#         self.in_planes, planes = c, [32, 64, 128, 256, 512]
#         share_planes = 8
#         stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
#         self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
#         self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
#         self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
#         self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
#         self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
#         self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
#         self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
#         self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
#         self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
#         self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
#         self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Dropout(p=dropout_p), nn.Linear(planes[0], k))
        

#     def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
#         layers = []
#         layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
#         self.in_planes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
#         return nn.Sequential(*layers)

#     def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
#         layers = []
#         layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
#         self.in_planes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
#         return nn.Sequential(*layers)

#     def forward(self, pxo):
#         p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
#         # 如果 self.c > 3，说明除了坐标还有额外特征，将其拼接
#         x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
#         p1, x1, o1 = self.enc1([p0, x0, o0])
#         p2, x2, o2 = self.enc2([p1, x1, o1])
#         p3, x3, o3 = self.enc3([p2, x2, o2])
#         p4, x4, o4 = self.enc4([p3, x3, o3])
#         p5, x5, o5 = self.enc5([p4, x4, o4])
#         x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
#         x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
#         x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
#         x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
#         x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
#         x = self.cls(x1)
#         # # 💡 新的输出逻辑
#         # feat = self.head_base(x1)
        
#         # out_recall = self.expert_recall(feat)
#         # out_refine = self.expert_refine(feat)
        
#         # # 计算门控权重
#         # gating_weights = self.gate(feat) # [N, 2]
        
#         # # 动态融合：w1 * expert1 + w2 * expert2
#         # x = gating_weights[:, 0:1] * out_recall + gating_weights[:, 1:2] * out_refine
#         return x

class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks=[2, 3, 3, 2], stride=None, nsample=None, share_planes=None, c=6, k=13, dropout_p=0):
        super().__init__()
        self.c = c
        self.num_stages = len(blocks)
        
        # 💡 1. 动态生成每层的特征维度、步长和采样点数
        # 维度以 32 起步，每次翻倍: [32, 64, 128, 256, 512, ...]
        planes = [32 * (2 ** i) for i in range(self.num_stages)] 
        # 步长: 第一层为 1，后续全是 4
        if stride is None:
            stride = [1] + [4] * (self.num_stages - 1)
        
        # 采样数: 第一层为 8，后续全是 16
        if nsample is None: 
            nsample = [8] + [16] * (self.num_stages - 1)
        
        
        self.in_planes = c
        if share_planes is None:
            share_planes = 8
        # ==========================================
        # 💡 2. 动态构建 Encoder (使用 nn.ModuleList)
        # ==========================================
        self.enc = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                self._make_enc(block, planes[i], blocks[i], share_planes, stride=stride[i], nsample=nsample[i])
            )
            
        # ==========================================
        # 💡 3. 动态构建 Decoder (使用 nn.ModuleList，倒序构建)
        # ==========================================
        self.dec = nn.ModuleList()
        for i in range(self.num_stages - 1, -1, -1):
            is_head = (i == self.num_stages - 1)
            # Decoder 的 block 数量默认固定为 2 (1个 TransitionUp + 1个 TransformerBlock)
            self.dec.append(
                self._make_dec(block, planes[i], 2, share_planes, nsample=nsample[i], is_head=is_head)
            )

        # 最终的分类/回归头
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]), 
            nn.BatchNorm1d(planes[0]), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=dropout_p), 
            nn.Linear(planes[0], k)
        )

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
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        
        curr_pxo = [p0, x0, o0]
        enc_outputs = []
        
        # ==========================================
        # 💡 4. 动态 Encoder 前向传播
        # ==========================================
        for enc_layer in self.enc:
            curr_pxo = enc_layer(curr_pxo)
            enc_outputs.append(curr_pxo)  # 存入列表，留给 Decoder 做特征融合
            
        # ==========================================
        # 💡 5. 动态 Decoder 前向传播
        # ==========================================
        # 获取最底层的特征 (对应原来的 p5, x5, o5)
        p_curr, x_curr, o_curr = enc_outputs[-1]
        
        for i, dec_layer in enumerate(self.dec):
            stage_idx = self.num_stages - 1 - i
            
            if i == 0: 
                # 最顶层的 Decoder (对应原来的 dec5，无 skip connection)
                x_curr = dec_layer[1:]([p_curr, dec_layer[0]([p_curr, x_curr, o_curr]), o_curr])[1]
            else:
                # 后续的 Decoder (对应原来的 dec4~dec1，需要通过列表读取前面 Encoder 的特征进行融合)
                p_skip, x_skip, o_skip = enc_outputs[stage_idx]
                x_curr = dec_layer[1:]([p_skip, dec_layer[0]([p_skip, x_skip, o_skip], [p_curr, x_curr, o_curr]), o_skip])[1]
                p_curr, o_curr = p_skip, o_skip
                
        # 分类头
        x = self.cls(x_curr)
        return x

# ==========================================
#  Part 2: 桥接 Wrapper (无缝替换旧模型)
# ==========================================

class get_model(nn.Module):
    def __init__(self, num_classes, input_dim=8, dropout_p=0, blocks=[2, 3, 4, 6, 3], stride=None, nsample=None, share_planes=None):
        super(get_model, self).__init__()
        self.input_dim = input_dim
        self.backbone = PointTransformerSeg(
            block=PointTransformerBlock, 
            blocks=blocks, 
            c=input_dim, 
            k=num_classes,
            dropout_p=dropout_p
        )

    # 💡 1. 接收 DataLoader 传来的 mask
    def forward(self, xyz, mask=None):
        """
        xyz: [B, input_dim, N] 
        mask: [B, N] (True 表示真实点, False 表示 Padding 点)
        """
        B, C_in, N = xyz.shape
        device = xyz.device
        
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        
        if mask is None:
            # 兼容旧代码，如果没有传mask，假定全是有效点
            mask = torch.ones((B, N), dtype=torch.bool, device=device)

        # ==========================================
        # 💡 核心魔法：数据浓缩 (剥离 Padding)
        # ==========================================
        # 直接利用 mask 提取所有真实点，打破 Batch 边界
        # valid_xyz 形状变为 [N_total_valid, C_in] (全 Batch 所有真实点的总和)
        valid_xyz = xyz_trans[mask] 
        
        p = valid_xyz[:, :3].contiguous()
        
        if self.input_dim > 3:
            x = valid_xyz[:, 3:].contiguous()
        else:
            x = None
            
        # 💡 重新构造真实的 Offset (o)
        # 统计每个 batch 里的有效点数量，并累加求和
        valid_counts = mask.sum(dim=1, dtype=torch.int32)
        o = torch.cumsum(valid_counts, dim=0).int().contiguous()
        
        # ==========================================
        # 送入骨干网络 (此时网络运算效率达 100%，没有任何废计算)
        # ==========================================
        out_valid = self.backbone([p, x, o])  # [N_total_valid, num_classes]
        
        # ==========================================
        # 💡 还原回规整的张量 (为 Loss 计算做准备)
        # ==========================================
        # 初始化一个全为极小值 (-1e4) 的张量
        # 为什么用极小值？因为经过 Sigmoid(-1e4) 后，Padding 点的预测概率会严格变为 0.0！
        out = torch.full((B, N, out_valid.shape[-1]), -1e4, device=device, dtype=out_valid.dtype)
        
        # 利用布尔索引，把算好的真实点精准塞回原本的位置
        out[mask] = out_valid 
        
        return out


# ==========================================
#  Part 3: 你的专属 Loss 模块 (完全保持原样)
# ==========================================

def focal_loss(logits, targets, weights=None, alpha=0.9, gamma=2.0, mask=None):
    
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
        
    # 💡 核心过滤：只对 mask 为 True 的有效点计算均值
    if mask is not None:
        loss = loss[mask]
        
    return loss.mean()

def centernet_focal_loss(logits, targets, weights=None, alpha=2.0, beta=4.0, mask=None, pos_weight=1.0):
    """
    仿照你的风格重写的 CenterNet Focal Loss
    :param alpha: 控制难易样本权重的超参数 (原版 Focal 的 gamma，推荐 2.0)
    :param beta: 💡 控制高斯拖尾区“宽容度”的超参数 (推荐 4.0)
    :param pos_weight: 正样本强心剂系数 (如果 mid 还是升不上去，可以改成 10.0)
    """
    # 1. 维度对齐处理
    logits = logits.squeeze(-1) if logits.dim() > 2 else logits
    targets = targets.squeeze(-1).float() if targets.dim() > 2 else targets
    # targets = torch.clamp((targets - 0.5) * 2.0, min=0.0, max=1.0)
    
    # 2. 概率化并安全截断 (非常关键，防止 log(0) 爆出 NaN)
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, min=1e-4, max=1.0 - 1e-4)
    
    # ==========================================
    # 💡 核心修改：分界条件弱化为 >= 0.9
    # ==========================================
    pos_mask = (targets >= 0.8).float()
    neg_mask = (targets < 0.8).float()
    
    # 3. 计算正样本 Loss (只看 >= 0.9 的点，逼迫它们输出 1.0)
    # 公式: -log(p) * (1-p)^alpha
    pos_loss = torch.log(probs) * torch.pow(1 - probs, alpha) * pos_mask * pos_weight
    
    # 4. 计算负样本 Loss (处理 < 0.9 的点，带免死金牌)
    # 💡 魔法系数 neg_weights：(1 - targets)^beta
    # 如果 targets 是 0.8，(1-0.8)^4 = 0.0016，乘在前面，惩罚直接免除 99%
    neg_weights = torch.pow(1 - targets, beta)
    neg_loss = torch.log(1 - probs) * torch.pow(probs, alpha) * neg_weights * neg_mask
    
    # 5. 汇总正负 Loss 
    loss = -(pos_loss + neg_loss)
    
    # 6. 附加外部权重
    if weights is not None:
        weights = weights.squeeze(-1) if weights.dim() > loss.dim() else weights
        loss = loss * weights
        
    # ==========================================
    # 💡 核心过滤与归一化
    # ==========================================
    if mask is not None:
        loss = loss[mask]
        pos_mask = pos_mask[mask]
        
    # CenterNet 的关键细节：除以正样本的数量，而不是单纯的 mean()
    # 如果用 mean()，1万个点里只有 2个正样本，正样本的梯度会被除以 10000，彻底消失
    num_pos = pos_mask.sum()
    
    if num_pos == 0:
        # 万一当前批次里(或者 mask 内)一个 >= 0.9 的点都没有，就用 mean 保底
        return loss.mean()
    else:
        # 把整个 mask 内的 Loss 总和，平摊到仅有的几个正样本头上
        return loss.sum() / num_pos


def get_skeleton_and_pairs(pb, xb, f_start_b, f_goal_b, delta_s=0.5, R_nms=0.5, K_local=3, delta_d=0.2, K_pairs=3):
    """
    K_pairs: 定义每个节点最多寻找的前向/后向邻居数量
    """
    # 💡 终极修复：利用特征场识别起终点
    is_start = f_start_b > 0.95
    is_goal = f_goal_b > 0.95
    
    # 💡 赋予免死金牌：只要预测分数达标，【或者】它是起终点，就允许进入候选池！
    base_mask = (pb > delta_s) | is_start | is_goal 
    
    idx_base = torch.nonzero(base_mask, as_tuple=False).squeeze(-1)
    
    if idx_base.numel() < 2:
        return None, None, None, None, None
        
    pb_base, xb_base = pb[idx_base], xb[idx_base]
    dist_base = torch.cdist(xb_base, xb_base)
    in_nms = dist_base < R_nms
    higher_p = pb_base.unsqueeze(0) < pb_base.unsqueeze(1) 
    higher_count = (in_nms & higher_p).sum(dim=1) 
    v_mask = higher_count < K_local
    V_idx = idx_base[v_mask]
    
    if V_idx.numel() < 2: 
        return None, None, None, None, None
        
    v_pb, v_xb = pb[V_idx], xb[V_idx]
    v_fs, v_fg = f_start_b[V_idx], f_goal_b[V_idx]
    dist_v = torch.cdist(v_xb, v_xb)
    
    # 💡 动态确定 Top-K 的真实数量 (防止筛选后存活的骨架点总数小于 K_pairs 导致报错)
    actual_K = min(K_pairs, V_idx.shape[0])
    
    # 找 k* 集合 (prev)
    cond_k_dist = dist_v > delta_d
    cond_k_fs = v_fs.unsqueeze(0) < v_fs.unsqueeze(1) 
    cond_k_fg = v_fg.unsqueeze(0) > v_fg.unsqueeze(1) 
    valid_k = cond_k_dist & cond_k_fs & cond_k_fg
    diff_fs = v_fs.unsqueeze(1) - v_fs.unsqueeze(0) 
    diff_fs = torch.where(valid_k, diff_fs, torch.full_like(diff_fs, float('inf')))
    
    # 💡 升级为集合：取差值最小的 Top-K 个邻居
    vals_k, k_star_local = torch.topk(diff_fs, k=actual_K, dim=1, largest=False)
    # 只要取出来的最小值不是 inf，就说明找到了合法的真实邻居，has_k 形状变为 [M, actual_K]
    has_k = (vals_k != float('inf'))
    
    # 找 j* 集合 (next)
    cond_j_dist = dist_v > delta_d
    cond_j_fg = v_fg.unsqueeze(0) < v_fg.unsqueeze(1) 
    cond_j_fs = v_fs.unsqueeze(0) > v_fs.unsqueeze(1) 
    valid_j = cond_j_dist & cond_j_fg & cond_j_fs
    diff_fg = v_fg.unsqueeze(1) - v_fg.unsqueeze(0) 
    diff_fg = torch.where(valid_j, diff_fg, torch.full_like(diff_fg, float('inf')))
    
    # 💡 升级为集合：取差值最小的 Top-K 个邻居
    vals_j, j_star_local = torch.topk(diff_fg, k=actual_K, dim=1, largest=False)
    # has_j 形状变为 [M, actual_K]
    has_j = (vals_j != float('inf'))
    
    return V_idx, k_star_local, j_star_local, has_k, has_j

def straightness_loss(p, xyz, f_start, f_goal, delta_s=0.5, R_nms=0.5, K_local=3, delta_d=0.2, K_pairs=3, r_corridor=0.03, rho=20000.0, alpha2=1.0, alpha3=1.0, tau_s=0.6, eps=1e-6):
    B, N, _ = xyz.shape
    device = xyz.device
    total_loss = []
    total_phi_s = [] # 💡 新增：记录 phi_s
    
    for b in range(B):
        pb, xb = p[b], xyz[b]
        f_start_b, f_goal_b = f_start[b], f_goal[b]
        
        V_idx, k_star, j_star, has_k, has_j = get_skeleton_and_pairs(
            pb, xb, f_start_b, f_goal_b, delta_s, R_nms, K_local, delta_d, K_pairs
        )
        if V_idx is None: continue
            
        v_xb, v_pb = xb[V_idx], pb[V_idx]
        v_fs, v_fg = f_start_b[V_idx], f_goal_b[V_idx]
        M = V_idx.shape[0]
        xb_all = xb 
        
        def get_best_q_straight(neighbors_idx, has_neighbors, is_prev=True):
            K_actual = neighbors_idx.shape[1]
            phi_s = torch.zeros((M, K_actual), device=device)
            phi_c = torch.zeros((M, K_actual), device=device)
            
            valid_mask = has_neighbors
            if not valid_mask.any():
                return torch.zeros(M, device=device), torch.zeros(M, dtype=torch.bool, device=device), torch.zeros(M, device=device)
                
            idx_m, idx_k = torch.nonzero(valid_mask, as_tuple=True)
            n_idx = neighbors_idx[idx_m, idx_k]
            p1, p2 = v_xb[idx_m], v_xb[n_idx]
            
            v = p2 - p1
            vv = (v**2).sum(-1, keepdim=True) + eps
            w = xb_all.unsqueeze(0) - p1.unsqueeze(1) 
            t = torch.clamp((w * v.unsqueeze(1)).sum(-1, keepdim=True) / vv.unsqueeze(1), 0.0, 1.0)
            proj = p1.unsqueeze(1) + t * v.unsqueeze(1)
            d_to_seg = torch.norm(xb_all.unsqueeze(0) - proj, dim=-1)
            
            N_ij = (d_to_seg <= r_corridor).sum(dim=-1).float()
            dist_ij = torch.norm(v, dim=-1)
            V_corridor = math.pi * (r_corridor**2) * dist_ij
            N_max = rho * V_corridor + eps
            phi_s[idx_m, idx_k] = torch.exp(alpha2 * (torch.clamp(N_ij / N_max, 0.0, 1.0) - 1.0))
            
            if is_prev: delta_f = v_fs[n_idx] - v_fs[idx_m]
            else: delta_f = v_fg[n_idx] - v_fg[idx_m]
            phi_c[idx_m, idx_k] = torch.exp(alpha3 * ((delta_f * delta_d) / (dist_ij + eps) - 1.0))
            
            p_x = v_pb[neighbors_idx]
            p_x = torch.where(has_neighbors, p_x, torch.zeros_like(p_x))
            
            Phi_total = phi_s * phi_c
            joint_score = p_x * Phi_total
            best_idx = torch.argmax(joint_score, dim=1)
            row_idx = torch.arange(M, device=device)
            
            best_phi_s = phi_s[row_idx, best_idx]
            best_p_x = p_x[row_idx, best_idx]
            has_any = has_neighbors.any(dim=1)
            
            Q_s = best_p_x * best_phi_s
            return Q_s, has_any, best_phi_s # 💡 新增返回 best_phi_s

        Q_s_k, has_any_k, phi_s_k = get_best_q_straight(k_star, has_k, is_prev=True)
        Q_s_j, has_any_j, phi_s_j = get_best_q_straight(j_star, has_j, is_prev=False)
        
        sum_Q = Q_s_k * has_any_k.float() + Q_s_j * has_any_j.float()
        sum_phi = phi_s_k * has_any_k.float() + phi_s_j * has_any_j.float() # 💡 累加 phi
        num_E = has_any_k.float() + has_any_j.float()
        
        valid_mask = num_E > 0
        if not valid_mask.any(): continue
            
        Q_straight = sum_Q[valid_mask] / num_E[valid_mask]
        mean_phi_s = (sum_phi[valid_mask] / num_E[valid_mask]).mean() # 💡 计算均值
        p_i_valid = v_pb[valid_mask]
        
        loss_b = (p_i_valid * (tau_s - Q_straight.detach())).sum() / valid_mask.sum()
        total_loss.append(loss_b)
        total_phi_s.append(mean_phi_s) # 💡 记录

    final_loss = torch.stack(total_loss).mean() if len(total_loss) > 0 else torch.tensor(0.0, device=device, requires_grad=True)
    final_phi = torch.stack(total_phi_s).mean() if len(total_phi_s) > 0 else torch.tensor(0.0, device=device)
    return final_loss, final_phi # 💡 返回双变量

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
        
        # ⚠️ 重要修正：计算 xb 中的高分点到【当前 batch 所有点云 xyz[b]】的距离
        # 而不是仅仅算 xb 到 xb 的距离。因为我们要统计的是真实环境的局部密度。
        dist = torch.cdist(xb, xyz[b]) 
        
        N_i = (dist <= r_local).sum(dim=-1).float()
        V_local = 4.0 / 3.0 * math.pi * r_local ** 3
        N_local_max = rho * V_local + eps
        
        # --- 核心修改逻辑 ---
        # 1. 计算比例
        ratio = N_i / N_local_max
        # 2. 加入截断，限制在 [0.0, 1.0]
        ratio = torch.clamp(ratio, min=0.0, max=1.0)
        # 3. 使用新公式，没有负号
        e_i = torch.exp(alpha1 * (ratio - 1.0))
        # --------------------
        
        # ⚠️ 重要修正：补充公式里的负号，指导优化器最大化这个得分
        loss_b = -1.0 * (pb * e_i.detach()).mean()
        losses.append(loss_b)
        
    if len(losses) == 0: 
        # 加上 requires_grad=True 防止极端情况下全被过滤导致反向传播报错
        return torch.tensor(0.0, device=device, requires_grad=True) 
        
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


def cost_loss(p, xyz, f_start, f_goal, delta_s=0.5, R_nms=0.5, K_local=3, delta_d=0.2, K_pairs=3, r_corridor=0.03, rho=20000.0, alpha2=1.0, alpha3=1.0, tau_c=0.6, eps=1e-6):
    B, N, _ = xyz.shape
    device = xyz.device
    total_loss = []
    total_phi_c = [] # 💡 新增
    
    for b in range(B):
        pb, xb = p[b], xyz[b]
        f_start_b, f_goal_b = f_start[b], f_goal[b]
        
        V_idx, k_star, j_star, has_k, has_j = get_skeleton_and_pairs(
            pb, xb, f_start_b, f_goal_b, delta_s, R_nms, K_local, delta_d, K_pairs
        )
        if V_idx is None: continue
            
        v_xb, v_pb = xb[V_idx], pb[V_idx]
        v_fs, v_fg = f_start_b[V_idx], f_goal_b[V_idx]
        M = V_idx.shape[0]
        xb_all = xb 
        
        def get_best_q_cost(neighbors_idx, has_neighbors, is_prev=True):
            K_actual = neighbors_idx.shape[1]
            phi_s = torch.zeros((M, K_actual), device=device)
            phi_c = torch.zeros((M, K_actual), device=device)
            
            valid_mask = has_neighbors
            if not valid_mask.any():
                return torch.zeros(M, device=device), torch.zeros(M, dtype=torch.bool, device=device), torch.zeros(M, device=device)
                
            idx_m, idx_k = torch.nonzero(valid_mask, as_tuple=True)
            n_idx = neighbors_idx[idx_m, idx_k]
            p1, p2 = v_xb[idx_m], v_xb[n_idx]
            
            v = p2 - p1
            vv = (v**2).sum(-1, keepdim=True) + eps
            w = xb_all.unsqueeze(0) - p1.unsqueeze(1) 
            t = torch.clamp((w * v.unsqueeze(1)).sum(-1, keepdim=True) / vv.unsqueeze(1), 0.0, 1.0)
            proj = p1.unsqueeze(1) + t * v.unsqueeze(1)
            d_to_seg = torch.norm(xb_all.unsqueeze(0) - proj, dim=-1)
            
            N_ij = (d_to_seg <= r_corridor).sum(dim=-1).float()
            dist_ij = torch.norm(v, dim=-1)
            V_corridor = math.pi * (r_corridor**2) * dist_ij
            N_max = rho * V_corridor + eps
            phi_s[idx_m, idx_k] = torch.exp(alpha2 * (torch.clamp(N_ij / N_max, 0.0, 1.0) - 1.0))
            
            if is_prev: delta_f = v_fs[n_idx] - v_fs[idx_m]
            else: delta_f = v_fg[n_idx] - v_fg[idx_m]
            phi_c[idx_m, idx_k] = torch.exp(alpha3 * ((delta_f * delta_d) / (dist_ij + eps) - 1.0))
            
            p_x = v_pb[neighbors_idx]
            p_x = torch.where(has_neighbors, p_x, torch.zeros_like(p_x))
            
            Phi_total = phi_s * phi_c
            joint_score = p_x * Phi_total
            best_idx = torch.argmax(joint_score, dim=1)
            row_idx = torch.arange(M, device=device)
            
            best_phi_c = phi_c[row_idx, best_idx]
            best_p_x = p_x[row_idx, best_idx]
            has_any = has_neighbors.any(dim=1)
            
            Q_c = best_p_x * best_phi_c
            return Q_c, has_any, best_phi_c # 💡 新增返回 best_phi_c

        Q_c_k, has_any_k, phi_c_k = get_best_q_cost(k_star, has_k, is_prev=True)
        Q_c_j, has_any_j, phi_c_j = get_best_q_cost(j_star, has_j, is_prev=False)
        
        sum_Q = Q_c_k * has_any_k.float() + Q_c_j * has_any_j.float()
        sum_phi = phi_c_k * has_any_k.float() + phi_c_j * has_any_j.float()
        num_E = has_any_k.float() + has_any_j.float()
        
        valid_mask = num_E > 0
        if not valid_mask.any(): continue
            
        Q_cost = sum_Q[valid_mask] / num_E[valid_mask]
        mean_phi_c = (sum_phi[valid_mask] / num_E[valid_mask]).mean()
        p_i_valid = v_pb[valid_mask]
        
        loss_b = (p_i_valid * (tau_c - Q_cost.detach())).sum() / valid_mask.sum()
        total_loss.append(loss_b)
        total_phi_c.append(mean_phi_c)

    final_loss = torch.stack(total_loss).mean() if len(total_loss) > 0 else torch.tensor(0.0, device=device, requires_grad=True)
    final_phi = torch.stack(total_phi_c).mean() if len(total_phi_c) > 0 else torch.tensor(0.0, device=device)
    return final_loss, final_phi

class get_loss(nn.Module):
    # 💡 修改 1: 移除了不再需要的 d_max=10.0
    # 💡 修改 2: 新增了 K_pairs=3
    # 💡 修改 3: 建议将 alpha3 默认值改为 1.0 (配合映射到 [e^-1, 1])
    # 💡 修改 4: 建议将 tau_c 默认值改为 0.6 (与 tau_s 保持合理的及格线)
    def __init__(self, w_bce=1.0, w_c_focal=0.0, w_straight=1.0, w_safety=0.2, w_conn=0.0, 
                 w_cost=1.0, alpha=0.6, gamma=2.0, delta_s=0.5, delta_d=0.2, 
                 R_nms=0.5, K_local=3, K_pairs=3, r_corridor=0.03, rho=20000.0, 
                 alpha2=1.0, alpha3=1.0, tau_s=0.6, tau_c=0.6, 
                 M_pair_max=256, r_local=0.05, alpha1=1.0, M_safe_max=256, 
                 delta_c=0.1, r_connect=0.05):
        super().__init__()
        # 权重设置
        self.w_bce = w_bce; self.w_c_focal=w_c_focal; self.w_straight = w_straight; self.w_safety = w_safety; self.w_conn = w_conn
        self.w_cost = w_cost
        
        # Focal Loss 参数
        self.alpha = alpha; self.gamma = gamma
        
        # 骨架提取与点对筛选参数
        self.delta_s = delta_s; self.delta_d = delta_d
        self.R_nms = R_nms; self.K_local = K_local
        self.K_pairs = K_pairs  # 集合最大容量
        
        # 物理评估参数
        self.r_corridor = r_corridor; self.rho = rho
        self.alpha2 = alpha2; self.alpha3 = alpha3
        
        # 动态奖惩基准线
        self.tau_s = tau_s; self.tau_c = tau_c
        
        # 其他安全与连通性参数
        self.M_pair_max = M_pair_max; self.r_local = r_local; self.alpha1 = alpha1
        self.M_safe_max = M_safe_max; self.delta_c = delta_c; self.r_connect = r_connect

    def forward(self, logits, targets, points, mask=None):
        if isinstance(logits, (tuple, list)): logits = logits[0]
        
        if points.shape[1] >= 5: 
            points_trans = points.permute(0, 2, 1).contiguous()
        else: 
            points_trans = points.contiguous()
            
        xyz_phys = points_trans[..., :3]
        f_start = points_trans[..., 3]
        f_goal = points_trans[..., 4]
        
        p = torch.sigmoid(logits)
        if p.dim() == 3: p = p.squeeze(-1)
        
        loss = 0.0
        metrics = {
            "loss_straight": 0.0,
            "loss_cost": 0.0,
            "phi_s": 0.0,
            "phi_c": 0.0
        }
        
        # 💡 3. 将 mask 传给 focal_loss
        loss += self.w_bce * focal_loss(logits, targets, weights=None, alpha=self.alpha, gamma=self.gamma, mask=mask)
        loss += self.w_c_focal * centernet_focal_loss(logits, targets, weights=None, alpha=2.0, beta=2.0, mask=mask, pos_weight=10.0)

        if self.w_straight > 0:
            l_str, p_s = straightness_loss(
                p=p, xyz=xyz_phys, f_start=f_start, f_goal=f_goal, 
                delta_s=self.delta_s, R_nms=self.R_nms, K_local=self.K_local, 
                delta_d=self.delta_d, K_pairs=self.K_pairs, 
                r_corridor=self.r_corridor, rho=self.rho, 
                alpha2=self.alpha2, alpha3=self.alpha3, tau_s=self.tau_s
            )
            loss += self.w_straight * l_str
            # 💡 填充字典
            metrics["loss_straight"] = l_str.item()
            metrics["phi_s"] = p_s.item()
            
        if self.w_safety > 0:
            loss += self.w_safety * safety_loss(
                p=p, xyz=xyz_phys, delta_s=self.delta_s, r_local=self.r_local, 
                rho=self.rho, alpha1=self.alpha1, M_max=self.M_safe_max
            )
            
        if self.w_conn > 0:
            loss += self.w_conn * connectivity_loss(
                p=p, xyz=xyz_phys, r_connect=self.r_connect, delta_c=self.delta_c
            )
            
        if self.w_cost > 0:
            l_cost, p_c = cost_loss(
                p=p, xyz=xyz_phys, f_start=f_start, f_goal=f_goal,
                delta_s=self.delta_s, R_nms=self.R_nms, K_local=self.K_local, 
                delta_d=self.delta_d, K_pairs=self.K_pairs, 
                r_corridor=self.r_corridor, rho=self.rho, 
                alpha2=self.alpha2, alpha3=self.alpha3, tau_c=self.tau_c
            )
            loss += self.w_cost * l_cost
            # 💡 填充字典
            metrics["loss_cost"] = l_cost.item()
            metrics["phi_c"] = p_c.item()

        return loss, metrics # 💡 返回元组 (总Loss, 监控字典)