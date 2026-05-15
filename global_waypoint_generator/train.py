import torch
import os
import time
import datetime
import glob
import numpy as np
import matplotlib.pyplot as plt

# 确保 DataLoader 路径与你的项目匹配
from src.data.data_loader import build_dataloader
from src.models.pointnet_transfomer2.my_model import get_model, get_loss, focal_loss

# --- Configuration ---
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# 1. 实验结果保存路径
SAVE_DIR = r"C:\Users\Administrator\Desktop\experiments\checkpoints"

# 2. 真实的训练数据路径
# DATA_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\src\data\data_for_train\train_data4"
DATA_DIR = r"C:\Users\Administrator\Desktop\experiments\train_data15"
# DATA_DIR = r"C:\Users\Administrator\Desktop\experiments\train_data5"

START_EPOCH = 1  # 如果从头训练填 1；如果调参直接从第 41 轮开始，填 41
PRETRAINED_CKPT = r"C:\Users\Administrator\Desktop\experiments\checkpoints\ckpt_epoch_40.pth" # 填入你第40轮保存的权重路径

TRAIN_STAGE = "A"  # 可选: "A" (训练管道) 或 "B" (训练航路点)
MODEL_A_CKPT = r"C:\Users\Administrator\Desktop\experiments\checkpoints\Stage_A\best_model.pth" # 阶段 B 需要用到 A 的权重

GAMMA = 2
ALPHA = 0.6
TOTAL_EPOCHS = 50
WARMUP_EPOCHS = 40    
# ---------------------      
# ---------------------

def _filter_stage_b_batch(points, targets_wp, valid_mask, probs_A, combined_mask, min_points=64):
    combined_counts = combined_mask.sum(dim=1)
    keep_samples = combined_counts >= min_points

    if keep_samples.all():
        return points, targets_wp, valid_mask, probs_A, combined_mask, False

    if not keep_samples.any():
        return None, None, None, None, None, True

    return (
        points[keep_samples],
        targets_wp[keep_samples],
        valid_mask[keep_samples],
        probs_A[keep_samples],
        combined_mask[keep_samples],
        False,
    )

def train_one_epoch_A(model, loader, criterion, optimizer, device, epoch_idx):
    # # --- Warm-up 策略保持原样 ---
    # if epoch_idx <= WARMUP_EPOCHS:
    #     criterion.w_bce = 20
    #     criterion.w_straight = 0.0
    #     criterion.w_cost = 0.0
    #     criterion.w_safety = 0.1  
    #     phase_name = "Warm-up (FOCAL Only)"
    # else:
    #     criterion.w_bce = 20
    #     criterion.w_straight = 5
    #     criterion.w_cost = 4
    #     criterion.w_safety = 0.1
    #     criterion.delta_s = 0.8
    #     criterion.delta_d = 0.2
    #     criterion.tau_s = 0.5  
    #     criterion.tau_c = 0.3  
    #     phase_name = "Refinement"

    model.train()
    total_loss = 0.0
    
    # --- Statistics Lists ---
    max_probs, neg_max_probs, neg_fpr_list = [], [], []
    se_mean_probs, mid_mean_probs, mid_fnr_list = [], [], []
    
    ep_l_str, ep_l_cost = [], []
    ep_phi_s, ep_phi_c = [], []

    # ==========================================
    # 💡 修改点 1：解包接收 mask
    # ==========================================
    for batch_idx, (points, targets, mask, gt_waypoints_list) in enumerate(loader):
        points = points.to(device)
        targets = targets[..., 0:1].to(device) # 只取管道真值
        mask = mask.to(device) # 💡 将 mask 也放入 GPU

        if points.shape[-1] >= 5:
            points = points.permute(0, 2, 1) 
        
        xyz = points[:, :3, :].permute(0, 2, 1).contiguous()


        optimizer.zero_grad()
        
        # ==========================================
        # 💡 修改点 2：把 mask 传给模型和 Loss
        # ==========================================
        output = model(torch.cat([points[:, :5, :], points[:, -3:-1, :]], dim=1), mask=mask) 
        logits = output[0] if isinstance(output, (tuple, list)) else output

        loss, metrics = criterion(logits, targets, points, mask=mask)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        ep_l_str.append(metrics["loss_straight"])
        ep_l_cost.append(metrics["loss_cost"])
        ep_phi_s.append(metrics["phi_s"])
        ep_phi_c.append(metrics["phi_c"])

        # ==========================================
        # 💡 修改点 3：统计指标时，使用 mask 过滤假点
        # ==========================================
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            if probs.dim() == 3: probs = probs.squeeze(-1)   # [B, N]
            if targets.dim() == 3: targets = targets.squeeze(-1) # [B, N]

            # 提取全图最大概率（仅限有效点）
            if mask.sum() > 0:
                max_probs.append(probs[mask].max().item())
            
            # 💡 负样本指标计算 (增加 & mask)
            neg_mask = (targets < 0.1) & mask 
            if neg_mask.sum() > 0:
                neg_probs = probs[neg_mask]
                neg_max_probs.append(neg_probs.max().item())
                num_false_pos = (neg_probs > 0.5).float().sum()
                fpr = num_false_pos / neg_probs.numel()
                neg_fpr_list.append(fpr.item())

            # 💡 正样本指标计算 (增加 & mask)
            pos_mask = (targets > 0.8) & mask
            if points.shape[1] >= 5:  
                f_start = points[:, 3, :]  
                f_goal  = points[:, 4, :]
                is_start_end = (f_start > 0.95) | (f_goal > 0.95)
                
                mask_SE = pos_mask & is_start_end       
                mask_MID = pos_mask & (~is_start_end)   

                if mask_SE.sum() > 0:
                    se_mean_probs.append(probs[mask_SE].mean().item())
                if mask_MID.sum() > 0:
                    mid_probs = probs[mask_MID]
                    mid_mean_probs.append(mid_probs.mean().item())
                    num_false_neg = (mid_probs < 0.5).float().sum()
                    fnr = num_false_neg / mid_probs.numel()
                    mid_fnr_list.append(fnr.item())

    # --- 后面的均值计算和 Print 保持原样 ---
    avg_neg_max = np.mean(neg_max_probs) if len(neg_max_probs) > 0 else 0.0
    avg_fpr = np.mean(neg_fpr_list) if len(neg_fpr_list) > 0 else 0.0
    avg_se_mean = np.mean(se_mean_probs) if len(se_mean_probs) > 0 else 0.0
    avg_mid_mean = np.mean(mid_mean_probs) if len(mid_mean_probs) > 0 else 0.0
    avg_mid_fnr = np.mean(mid_fnr_list) if len(mid_fnr_list) > 0 else 0.0  
    
    avg_l_str = np.mean(ep_l_str) if len(ep_l_str) > 0 else 0.0
    avg_l_cost = np.mean(ep_l_cost) if len(ep_l_cost) > 0 else 0.0
    avg_phi_s = np.mean(ep_phi_s) if len(ep_phi_s) > 0 else 0.0
    avg_phi_c = np.mean(ep_phi_c) if len(ep_phi_c) > 0 else 0.0

    print(f"[Debug Ep{epoch_idx}] |\n"
          f"  > Probs: SE:{avg_se_mean:.3f} | Mid:{avg_mid_mean:.3f} | NegMax:{avg_neg_max:.3f} | FPR:{avg_fpr:.4f} | FNR:{avg_mid_fnr:.4f}\n"
          f"  > Graph: Phi_s:{avg_phi_s:.3f} | Phi_c:{avg_phi_c:.3f} | L_str:{avg_l_str:.4f} | L_cost:{avg_l_cost:.4f}") 

    return total_loss / len(loader)

def train_one_epoch_B(model_B, model_A, loader, criterion, optimizer, device, epoch_idx, tube_thresh=0.4):
    """
    航路点网络 (Model B) 的专属训练循环
    参数说明:
    - model_B: 正在训练的航路点网络
    - model_A: 已经训练好并冻结的管道网络 (用于提供先验特征和截流掩码)
    - tube_thresh: 管道掩码的宽容截流阈值 (建议 0.05 ~ 0.1，保证不断连)
    """
    model_B.train()
    model_A.eval() # 💡 管道网络必须在 eval 模式，且不更新梯度
    
    total_loss = 0.0
    processed_batches = 0
    
    # --- Statistics Lists ---
    max_probs, neg_max_probs, neg_fpr_list = [], [], []
    se_mean_probs, mid_mean_probs, mid_fnr_list = [], [], []
    
    ep_l_str, ep_l_cost = [], []
    ep_phi_s, ep_phi_c = [], []

    for batch_idx, (points, targets, valid_mask, gt_waypoints_list) in enumerate(loader):
        points = points.to(device)
        
        # ==========================================
        # 💡 级联改动 1：切片获取航路点真值 (通道 1)
        # ==========================================
        targets_wp = targets[..., 1:2].to(device) 
        valid_mask = valid_mask.to(device)

        # 调整特征维度使其符合 Point Transformer 习惯 (通常是 [B, D, N])
        if points.shape[-1] >= 5:
            points = points.permute(0, 2, 1) 
        
        xyz = points[:, :3, :].permute(0, 2, 1).contiguous()

        # ==========================================
        # 💡 级联改动 2：无梯度运行 Model A，获取软先验与管道掩码
        # ==========================================
        with torch.no_grad():
            output_A = model_A(points, mask=valid_mask)
            logits_A = output_A[0] if isinstance(output_A, (tuple, list)) else output_A
            probs_A = torch.sigmoid(logits_A) # [B, 1, N] 或 [B, N, 1]
            
            # 生成管道掩码 (比如 > 0.1 认为是管内)
            # 注意 squeeze 保证维度和 valid_mask [B, N] 对齐
            tube_mask = (probs_A > tube_thresh).squeeze(-1) if probs_A.dim() == 3 else (probs_A > tube_thresh).squeeze()

        # ==========================================
        # 💡 级联改动 3：合并掩码 (Padding掩码 AND 管道掩码)
        # 只有既是真实点，又在管道内的点，才参与后续计算和 Loss 惩罚
        # ==========================================
        combined_mask = valid_mask & tube_mask
        points, targets_wp, valid_mask, probs_A, combined_mask, should_skip = _filter_stage_b_batch(
            points, targets_wp, valid_mask, probs_A, combined_mask, min_points=64
        )
        if should_skip:
            continue

        # ==========================================
        # 💡 级联改动 4：特征拼接 (原始特征 + 管道概率)
        # ==========================================
        # 根据你的 points 维度进行对齐拼接
        # 如果 points 是 [B, D, N]，probs_A 是 [B, 1, N]
        if probs_A.dim() == 3:
            if probs_A.shape[1] != 1:  # 如果它是 [B, N, 1]
                probs_A = probs_A.permute(0, 2, 1) # 翻转为 [B, 1, N]
        elif probs_A.dim() == 2:       # 如果它是 [B, N]
            probs_A = probs_A.unsqueeze(1)         # 增加通道维变成 [B, 1, N]
            
        # 2. 完美拼接！在通道维度 (dim=1) 把 9 个特征和 1 个概率拼起来
        points_with_prior = torch.cat([points, probs_A], dim=1) # 得到 [B, 10, N]
        points_with_prior = torch.cat([points[:, :4, :]], dim=1)

        # ------------------------------------------
        # 开始训练 Model B
        # ------------------------------------------
        optimizer.zero_grad()
        
        # 💡 将拼接后的特征 和 联合掩码 喂给 Model B
        output_B = model_B(points_with_prior, mask=combined_mask) 
        logits_B = output_B[0] if isinstance(output_B, (tuple, list)) else output_B

        # 💡 Loss 计算同样使用 联合掩码 combined_mask
        loss, metrics = criterion(logits_B, targets_wp, points_with_prior, mask=combined_mask)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        processed_batches += 1
        
        ep_l_str.append(metrics.get("loss_straight", 0.0))
        ep_l_cost.append(metrics.get("loss_cost", 0.0))
        ep_phi_s.append(metrics.get("phi_s", 0.0))
        ep_phi_c.append(metrics.get("phi_c", 0.0))

        # ==========================================
        # 💡 级联改动 5：统计指标时，全部基于 combined_mask 进行过滤
        # ==========================================
        with torch.no_grad():
            probs_B = torch.sigmoid(logits_B)
            if probs_B.dim() == 3: probs_B = probs_B.squeeze()   # [B, N]
            if targets_wp.dim() == 3: targets_wp = targets_wp.squeeze() # [B, N]

            # 提取管内最大概率
            if combined_mask.sum() > 0:
                max_probs.append(probs_B[combined_mask].max().item())
            
            # 💡 负样本指标计算 (基于高斯真值 < 0.5 且 在管内)
            neg_mask = (targets_wp < 0.1) & combined_mask 
            if neg_mask.sum() > 0:
                neg_probs = probs_B[neg_mask]
                neg_max_probs.append(neg_probs.max().item())
                num_false_pos = (neg_probs > 0.5).float().sum()
                fpr = num_false_pos / neg_probs.numel()
                neg_fpr_list.append(fpr.item())

            # 💡 正样本指标计算 (基于高斯真值 > 0.8 且 在管内)
            pos_mask = (targets_wp > 0.8) & combined_mask
            
            # (注意: 这里恢复原始的 points[:, :3, :] 去取几何坐标，因为 points_with_prior 包含了概率)
            if points.shape[1] >= 5:  
                f_start = points[:, 3, :]  
                f_goal  = points[:, 4, :]
                is_start_end = (f_start > 0.95) | (f_goal > 0.95)
                
                mask_SE = pos_mask & is_start_end       
                mask_MID = pos_mask & (~is_start_end)   

                if mask_SE.sum() > 0:
                    se_mean_probs.append(probs_B[mask_SE].mean().item())
                if mask_MID.sum() > 0:
                    mid_probs = probs_B[mask_MID]
                    mid_mean_probs.append(mid_probs.mean().item())
                    num_false_neg = (mid_probs < 0.5).float().sum()
                    fnr = num_false_neg / mid_probs.numel()
                    mid_fnr_list.append(fnr.item())

    # --- 均值计算与打印 ---
    avg_neg_max = np.mean(neg_max_probs) if len(neg_max_probs) > 0 else 0.0
    avg_fpr = np.mean(neg_fpr_list) if len(neg_fpr_list) > 0 else 0.0
    avg_se_mean = np.mean(se_mean_probs) if len(se_mean_probs) > 0 else 0.0
    avg_mid_mean = np.mean(mid_mean_probs) if len(mid_mean_probs) > 0 else 0.0
    avg_mid_fnr = np.mean(mid_fnr_list) if len(mid_fnr_list) > 0 else 0.0  
    
    avg_l_str = np.mean(ep_l_str) if len(ep_l_str) > 0 else 0.0
    avg_l_cost = np.mean(ep_l_cost) if len(ep_l_cost) > 0 else 0.0
    avg_phi_s = np.mean(ep_phi_s) if len(ep_phi_s) > 0 else 0.0
    avg_phi_c = np.mean(ep_phi_c) if len(ep_phi_c) > 0 else 0.0

    print(f"[Model B - Ep{epoch_idx}] |\n"
          f"  > Probs: SE:{avg_se_mean:.3f} | Mid:{avg_mid_mean:.3f} | NegMax:{avg_neg_max:.3f} | FPR:{avg_fpr:.4f} | FNR:{avg_mid_fnr:.4f}\n"
          f"  > Graph: Phi_s:{avg_phi_s:.3f} | Phi_c:{avg_phi_c:.3f} | L_str:{avg_l_str:.4f} | L_cost:{avg_l_cost:.4f}") 

    return total_loss / max(processed_batches, 1)
class EarlyStopping:
    def __init__(self, patience=15, delta=0.001, save_dir='checkpoints'):
        """
        patience: 容忍多少个 epoch 验证集 loss 不下降
        delta: loss 至少要下降多少才算真的下降
        save_dir: 权重保存的文件夹路径
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.save_dir = save_dir

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            # 创新低，保存最优模型，清空计数器
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
            print(f"  >>> New Best Model Saved! (Val Loss: {val_loss:.4f})")
        else:
            # 没创新低，计数器加 1
            self.counter += 1
            print(f"  >>> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

@torch.no_grad()
def validate_A(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    # --- Statistics Lists (与训练集一致) ---
    max_probs, neg_max_probs, neg_fpr_list = [], [], []
    se_mean_probs, mid_mean_probs, mid_fnr_list = [], [], []
    
    ep_l_str, ep_l_cost = [], []
    ep_phi_s, ep_phi_c = [], []

    # 💡 验证阶段全过程关闭梯度计算，节省显存
    with torch.no_grad():
        # 💡 1. 接收 mask
        for points, targets, mask, gt_waypoints_list in loader:
            points = points.to(device)
            targets = targets[..., 0:1].to(device)
            mask = mask.to(device) # 💡 将 mask 放入 GPU
            
            if points.shape[-1] >= 5:
                points = points.permute(0, 2, 1)
                
            xyz = points[:, :3, :].permute(0, 2, 1).contiguous()

            # 💡 2. 传入 mask 给模型
            output = model(torch.cat([points[:, :5, :], points[:, -3:-1, :]], dim=1), mask=mask)
            logits = output[0] if isinstance(output, (tuple, list)) else output

            # 💡 3. 传入 mask 给 criterion，并接收 metrics 进行统计
            loss, metrics = criterion(logits, targets, points, mask=mask)
            total_loss += loss.item()

            # --- 记录 Graph 相关指标 ---
            ep_l_str.append(metrics["loss_straight"])
            ep_l_cost.append(metrics["loss_cost"])
            ep_phi_s.append(metrics["phi_s"])
            ep_phi_c.append(metrics["phi_c"])

            # --- 统计指标，使用 mask 过滤假点 ---
            probs = torch.sigmoid(logits)
            if probs.dim() == 3: probs = probs.squeeze(-1)   # [B, N]
            if targets.dim() == 3: targets = targets.squeeze(-1) # [B, N]

            # 提取全图最大概率（仅限有效点）
            if mask.sum() > 0:
                max_probs.append(probs[mask].max().item())
            
            # 💡 负样本指标计算
            neg_mask = (targets < 0.1) & mask 
            if neg_mask.sum() > 0:
                neg_probs = probs[neg_mask]
                neg_max_probs.append(neg_probs.max().item())
                num_false_pos = (neg_probs > 0.5).float().sum()
                fpr = num_false_pos / neg_probs.numel()
                neg_fpr_list.append(fpr.item())

            # 💡 正样本指标计算
            pos_mask = (targets > 0.8) & mask
            if points.shape[1] >= 5:  
                f_start = points[:, 3, :]  
                f_goal  = points[:, 4, :]
                is_start_end = (f_start > 0.95) | (f_goal > 0.95)
                
                mask_SE = pos_mask & is_start_end       
                mask_MID = pos_mask & (~is_start_end)   

                if mask_SE.sum() > 0:
                    se_mean_probs.append(probs[mask_SE].mean().item())
                if mask_MID.sum() > 0:
                    mid_probs = probs[mask_MID]
                    mid_mean_probs.append(mid_probs.mean().item())
                    num_false_neg = (mid_probs < 0.5).float().sum()
                    fnr = num_false_neg / mid_probs.numel()
                    mid_fnr_list.append(fnr.item())

    # --- 后面的均值计算和 Print ---
    avg_neg_max = np.mean(neg_max_probs) if len(neg_max_probs) > 0 else 0.0
    avg_fpr = np.mean(neg_fpr_list) if len(neg_fpr_list) > 0 else 0.0
    avg_se_mean = np.mean(se_mean_probs) if len(se_mean_probs) > 0 else 0.0
    avg_mid_mean = np.mean(mid_mean_probs) if len(mid_mean_probs) > 0 else 0.0
    avg_mid_fnr = np.mean(mid_fnr_list) if len(mid_fnr_list) > 0 else 0.0  
    
    avg_l_str = np.mean(ep_l_str) if len(ep_l_str) > 0 else 0.0
    avg_l_cost = np.mean(ep_l_cost) if len(ep_l_cost) > 0 else 0.0
    avg_phi_s = np.mean(ep_phi_s) if len(ep_phi_s) > 0 else 0.0
    avg_phi_c = np.mean(ep_phi_c) if len(ep_phi_c) > 0 else 0.0

    # 打印前缀改为 [Validation] 用于区分训练输出
    print(f"[Validation] |\n"
          f"  > Probs: SE:{avg_se_mean:.3f} | Mid:{avg_mid_mean:.3f} | NegMax:{avg_neg_max:.3f} | FPR:{avg_fpr:.4f} | FNR:{avg_mid_fnr:.4f}\n"
          f"  > Graph: Phi_s:{avg_phi_s:.3f} | Phi_c:{avg_phi_c:.3f} | L_str:{avg_l_str:.4f} | L_cost:{avg_l_cost:.4f}") 

    return total_loss / len(loader)

def validate_B(model_B, model_A, loader, criterion, device, tube_thresh=0.4):
    """
    航路点网络 (Model B) 的验证循环
    """
    # 两个模型都必须处于 eval 模式
    model_B.eval()
    model_A.eval() 
    
    total_loss = 0.0
    processed_batches = 0

    # --- 用于存放验证集检测指标的列表 ---
    neg_max_probs, neg_fpr_list = [], []
    se_mean_probs, mid_mean_probs, mid_fnr_list = [], [], []

    # 💡 验证阶段全程不需要计算梯度，极大节省显存并加速
    with torch.no_grad():
        for points, targets, valid_mask, gt_waypoints_list in loader:
            points = points.to(device)
            valid_mask = valid_mask.to(device)
            
            # 💡 1. 验证集也需要切片，只取航路点真值 (通道 1)
            targets_wp = targets[..., 1:2].to(device) 
            
            if points.shape[-1] >= 5:
                points = points.permute(0, 2, 1)

            # 💡 2. Model A 提取软先验与管道掩码
            output_A = model_A(points, mask=valid_mask)
            logits_A = output_A[0] if isinstance(output_A, (tuple, list)) else output_A
            probs_A = torch.sigmoid(logits_A)
            
            tube_mask = (probs_A > tube_thresh).squeeze(-1) if probs_A.dim() == 3 else (probs_A > tube_thresh).squeeze()

            # 💡 3. 合并掩码 (有效点 AND 管内点)
            combined_mask = valid_mask & tube_mask
            points, targets_wp, valid_mask, probs_A, combined_mask, should_skip = _filter_stage_b_batch(
                points, targets_wp, valid_mask, probs_A, combined_mask, min_points=64
            )
            if should_skip:
                continue

            # 💡 4. 特征拼接 (对齐维度拼接)
            if probs_A.dim() == 3:
                if probs_A.shape[1] != 1:
                    probs_A = probs_A.permute(0, 2, 1)
            elif probs_A.dim() == 2:
                probs_A = probs_A.unsqueeze(1)
            
            # 特征消融：拼接前 5 个特征和最后 3 个特征（带先验）
            points_with_prior = torch.cat([points, probs_A], dim=1)
            points_with_prior = torch.cat([points[:, :4, :]], dim=1)

            # 💡 5. 传入联合掩码和新特征给 Model B
            output_B = model_B(points_with_prior, mask=combined_mask)
            logits_B = output_B[0] if isinstance(output_B, (tuple, list)) else output_B

            # 💡 6. 用联合掩码计算 Validation Loss
            loss, _ = criterion(logits_B, targets_wp, points_with_prior, mask=combined_mask)
            total_loss += loss.item()
            processed_batches += 1

            # ==========================================
            # 💡 7. 简洁检测指标统计 (Probs, FPR, FNR)
            # ==========================================
            probs_B = torch.sigmoid(logits_B)
            if probs_B.dim() == 3: probs_B = probs_B.squeeze()
            if targets_wp.dim() == 3: targets_wp = targets_wp.squeeze()

            # 负样本统计 (< 0.1)
            neg_mask = (targets_wp < 0.1) & combined_mask 
            if neg_mask.sum() > 0:
                neg_probs = probs_B[neg_mask]
                neg_max_probs.append(neg_probs.max().item())
                num_false_pos = (neg_probs > 0.5).float().sum()
                fpr = num_false_pos / neg_probs.numel()
                neg_fpr_list.append(fpr.item())

            # 正样本统计 (> 0.8)
            pos_mask = (targets_wp > 0.8) & combined_mask
            if points.shape[1] >= 5:  
                f_start = points[:, 3, :]  
                f_goal  = points[:, 4, :]
                is_start_end = (f_start > 0.95) | (f_goal > 0.95)
                
                mask_SE = pos_mask & is_start_end       
                mask_MID = pos_mask & (~is_start_end)   

                if mask_SE.sum() > 0:
                    se_mean_probs.append(probs_B[mask_SE].mean().item())
                if mask_MID.sum() > 0:
                    mid_probs = probs_B[mask_MID]
                    mid_mean_probs.append(mid_probs.mean().item())
                    num_false_neg = (mid_probs < 0.5).float().sum()
                    fnr = num_false_neg / mid_probs.numel()
                    mid_fnr_list.append(fnr.item())

    # --- 均值计算与简洁打印 ---
    avg_loss = total_loss / max(processed_batches, 1)
    
    avg_neg_max = np.mean(neg_max_probs) if len(neg_max_probs) > 0 else 0.0
    avg_fpr = np.mean(neg_fpr_list) if len(neg_fpr_list) > 0 else 0.0
    avg_se_mean = np.mean(se_mean_probs) if len(se_mean_probs) > 0 else 0.0
    avg_mid_mean = np.mean(mid_mean_probs) if len(mid_mean_probs) > 0 else 0.0
    avg_mid_fnr = np.mean(mid_fnr_list) if len(mid_fnr_list) > 0 else 0.0  
    
    print(f"[Validation B] : "
          f"SE:{avg_se_mean:.3f} | Mid:{avg_mid_mean:.3f} | NegMax:{avg_neg_max:.3f} | FPR:{avg_fpr:.4f} | FNR:{avg_mid_fnr:.4f}") 

    return avg_loss

def plot_loss_curve(train_loss, val_loss, save_dir):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 区分保存路径，防止 A 和 B 的权重互相覆盖
    save_dir = os.path.join(SAVE_DIR, f"Stage_{TRAIN_STAGE}") 
    os.makedirs(save_dir, exist_ok=True)
    print(f"[*] Checkpoints will be saved to: {save_dir}")

    # ==========================================
    # 2. Data Loading (完全复用)
    # ==========================================
    data_dir = DATA_DIR
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if len(all_files) == 0:
        print(f"[!] Error: No .npz files found in {data_dir}")
        return

    split = int(len(all_files) * 0.8)
    train_files = all_files[:split]
    val_files = all_files[split:]

    train_loader = build_dataloader(train_files, batch_size=32, shuffle=True)
    val_loader   = build_dataloader(val_files, batch_size=32, shuffle=False)

    # --- 动态探针 ---
    sample_points, _, _, _ = next(iter(train_loader)) # 记得解包 4 个变量哦
    real_input_dim = sample_points.shape[-1]
    print(f"[*] 动态检测到原始数据特征维度为: {real_input_dim}")

    # =====================================================================
    # 🚀 阶段 A：只训练管道模型 (Model A)
    # =====================================================================
    if TRAIN_STAGE == "A":
        print("\n" + "="*50)
        print("🚀 启动阶段一：训练管道探路模型 (Model A)")
        print("="*50)
        
        model_A = get_model(num_classes=1, input_dim=7, dropout_p=0.0).to(device)
        
        # ... (这里保留你原本的 Model A 预训练加载逻辑和 loss 配置) ...
        criterion = get_loss(
                            w_bce=20.0,
                            w_straight=0.0,
                            w_safety=0.0,
                            w_conn=0.0,
                            w_cost=0.0,
                            alpha=ALPHA,
                            gamma=GAMMA,
                            delta_s=0.6,
                            delta_d=0.2,
                            r_corridor=0.05,
                            r_local=0.06,
                            rho=25600.0,
                            # 💡 --- 新增的点对筛选与奖惩参数 ---
                            R_nms=0.15,       # 骨架点 NMS 抑制半径
                            K_local=3,
                            K_pairs=3,       # 局部允许的候选点最大数量 (增加多样性)
                            alpha2=2.0,      # 直线 Loss 的指数敏感度
                            alpha3=1,      # 成本 Loss 的指数敏感度
                            tau_s=0,       # 直线通畅度及格线 (大于奖励，小于惩罚)
                            tau_c=0,       # 成本(步长)及格线
                        ).to(device)
        
        optimizer = torch.optim.Adam(model_A.parameters(), lr=1e-4, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        early_stopping = EarlyStopping(patience=10, delta=0.001, save_dir=save_dir)

        train_loss_list, val_loss_list = [], []

        for epoch in range(START_EPOCH, TOTAL_EPOCHS + 1):
            train_loss = train_one_epoch_A(model_A, train_loader, criterion, optimizer, device, epoch)
            val_loss = validate_A(model_A, val_loader, criterion, device)
            
            scheduler.step()
            
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            
            print(f"[Epoch {epoch:03d}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
            torch.save(model_A.state_dict(), os.path.join(save_dir, "last_model_A.pth"))
            early_stopping(val_loss, model_A)
            
            if early_stopping.early_stop:
                print("🛑 触发早停，阶段 A 训练结束。")
                break


    # =====================================================================
    # 🎯 阶段 B：级联训练航路点模型 (Model B)
    # =====================================================================
    elif TRAIN_STAGE == "B":
        print("\n" + "="*50)
        print("🎯 启动阶段二：级联训练航路点模型 (Model B)")
        print("="*50)
        
        # 💡 1. 实例化并绝对冻结 Model A
        model_A = get_model(num_classes=1, input_dim=real_input_dim, dropout_p=0.2).to(device)
        if os.path.exists(MODEL_A_CKPT):
            # 1. 先把权重读取到内存字典中
            checkpoint = torch.load(MODEL_A_CKPT, map_location=device)
            new_state_dict = {}
            
            # 2. 遍历字典，翻译 Key 的名字
            for k, v in checkpoint.items():
                # 替换 Encoder 名称
                if 'enc1.' in k: k = k.replace('enc1.', 'enc.0.')
                elif 'enc2.' in k: k = k.replace('enc2.', 'enc.1.')
                elif 'enc3.' in k: k = k.replace('enc3.', 'enc.2.')
                elif 'enc4.' in k: k = k.replace('enc4.', 'enc.3.')
                elif 'enc5.' in k: k = k.replace('enc5.', 'enc.4.')
                
                # 替换 Decoder 名称
                # 注意：我们在 __init__ 中是用 range(4, -1, -1) 倒序生成 dec 的
                # 所以原来的 dec5 变成了现在的 dec.0，dec4 变成了 dec.1，以此类推
                elif 'dec5.' in k: k = k.replace('dec5.', 'dec.0.')
                elif 'dec4.' in k: k = k.replace('dec4.', 'dec.1.')
                elif 'dec3.' in k: k = k.replace('dec3.', 'dec.2.')
                elif 'dec2.' in k: k = k.replace('dec2.', 'dec.3.')
                elif 'dec1.' in k: k = k.replace('dec1.', 'dec.4.')
                
                new_state_dict[k] = v
                
            # 3. 将翻译好的新字典喂给模型
            model_A.load_state_dict(new_state_dict)
            print(f"[*] ✅ 成功通过字典映射加载 Model A 管道先验权重!")
            
        model_A.eval() # 锁定 Dropout 和 BatchNorm
        for param in model_A.parameters():
            param.requires_grad = False # 彻底切断梯度，省下海量显存！
            
        # 💡 2. 实例化 Model B (核心：输入维度 + 1，因为拼接了 P_tube)
        # 注意：这里你可以把 dropout 调高一点，防止在小规模正样本上过拟合
        # 测试发现share_planes比较大时效果更好，stride保证最终下采样层的点数在32-128区间，nsample貌似没有什么影响
        model_B = get_model(num_classes=1, input_dim=4, dropout_p=0.1, blocks=[2,4,2,2], stride=[1,4,2,2], nsample=[8,16,16,16], share_planes=16).to(device)
        print(f"[*] 🚀 Model B 已初始化，输入特征维度已自动扩展至: {real_input_dim + 1}")

        # 💡 3. Model B 专属的 loss
        criterion_B = get_loss(
                            w_bce=20.0,
                            w_c_focal=0.0,
                            w_straight=0.0,
                            w_safety=0.0,
                            w_conn=0.0,
                            w_cost=0.0,
                            alpha=0.9,
                            gamma=2.0,
                            delta_s=0.7,
                            delta_d=0.2,
                            r_corridor=0.05,
                            r_local=0.06,
                            rho=25600.0,
                            # 💡 --- 新增的点对筛选与奖惩参数 ---
                            R_nms=0.15,       # 骨架点 NMS 抑制半径
                            K_local=3,
                            K_pairs=3,       # 局部允许的候选点最大数量 (增加多样性)
                            alpha2=2.0,      # 直线 Loss 的指数敏感度
                            alpha3=1,      # 成本 Loss 的指数敏感度
                            tau_s=0,       # 直线通畅度及格线 (大于奖励，小于惩罚)
                            tau_c=0,       # 成本(步长)及格线
                        ).to(device) # 直接使用函数，或者用你自己封装的类
        
        optimizer_B = torch.optim.Adam(model_B.parameters(), lr=1e-4, weight_decay=5e-5)
        scheduler_B = torch.optim.lr_scheduler.StepLR(optimizer_B, step_size=15, gamma=0.5)
        early_stopping_B = EarlyStopping(patience=10, delta=0.001, save_dir=save_dir)

        train_loss_list, val_loss_list = [], []

        for epoch in range(START_EPOCH, TOTAL_EPOCHS + 1):
            # 💡 调用专属的 train_one_epoch_B
            train_loss = train_one_epoch_B(
                model_B=model_B, 
                model_A=model_A, 
                loader=train_loader, 
                criterion=criterion_B, 
                optimizer=optimizer_B, 
                device=device, 
                epoch_idx=epoch, 
                tube_thresh=0.4 # 宽容截流阈值
            )
            
            # 💡 调用专属的 validate_B
            val_loss = validate_B(
                model_B=model_B, 
                model_A=model_A, 
                loader=val_loader, 
                criterion=criterion_B, 
                device=device,
                tube_thresh=0.4
            )
            
            scheduler_B.step()
            current_lr = scheduler_B.get_last_lr()[0]
            
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            print(f"[Epoch {epoch:03d}/{TOTAL_EPOCHS}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

            torch.save(model_B.state_dict(), os.path.join(save_dir, "last_model_B.pth"))
            early_stopping_B(val_loss, model_B)
            
            if early_stopping_B.early_stop:
                print(f"🛑 触发早停机制！Model B 提前结束训练。")
                break

    print("\n[*] 训练结束，绘制最终 Loss 曲线...")
    plot_loss_curve(train_loss_list, val_loss_list, save_dir)
    
if __name__ == "__main__":
    print("开始计时...")
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_time = end_time - start_time
    time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f"\n{'='*40}")
    print(f"训练全部结束！总耗时: {time_str}")
    print(f"{'='*40}")
