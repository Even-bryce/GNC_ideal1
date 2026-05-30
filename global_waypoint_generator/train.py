import torch
import os
import time
import datetime
import glob
from tqdm import tqdm
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
DATA_DIR = r"C:\Users\Administrator\Desktop\experiments\train_data66"
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
        points_input = points[:, :4, :]


        optimizer.zero_grad()
        
        # ==========================================
        # 💡 修改点 2：把 mask 传给模型和 Loss
        # ==========================================
        output = model(points_input, mask=mask) 
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
                    num_false_neg = (mid_probs < 0.4).float().sum()
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

    # ==========================================
    # 💡 新增：用于统计 Model A 数据完备性（召回率）的计数器
    # ==========================================
    total_gt_high_conf = 0    # 真值 > 0.8 的总点数
    kept_by_model_A = 0       # 这些点中，被 Model A 成功保留的点数

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
            tube_mask = (probs_A > tube_thresh).squeeze(-1) if probs_A.dim() == 3 else (probs_A > tube_thresh).squeeze()

        # ==========================================
        # 💡 新增：检查 Model A 对高置信度真值的覆盖率 (完备性检查)
        # 必须在 _filter_stage_b_batch 之前进行，以免点数被裁剪
        # ==========================================
        with torch.no_grad():
            # 使用 reshape(-1) 展平为一维张量，避免 batch_size=1 时 squeeze 的不可预料行为
            _valid_flat = valid_mask.reshape(-1)
            _targets_flat = targets_wp.reshape(-1) 
            _probs_A_flat = probs_A.reshape(-1)
            
            # 找到所有 padding 内（有效）且真值 > 0.8 的点
            gt_high_conf_mask = (_targets_flat > 0.8) & _valid_flat
            
            # 在这些高置信度点中，看有多少概率大于 tube_thresh
            model_A_kept_mask = gt_high_conf_mask & (_probs_A_flat > tube_thresh)
            
            # 累加到全局统计
            total_gt_high_conf += gt_high_conf_mask.sum().item()
            kept_by_model_A += model_A_kept_mask.sum().item()

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
        if probs_A.dim() == 3:
            if probs_A.shape[1] != 1:  # 如果它是 [B, N, 1]
                probs_A = probs_A.permute(0, 2, 1) # 翻转为 [B, 1, N]
        elif probs_A.dim() == 2:       # 如果它是 [B, N]
            probs_A = probs_A.unsqueeze(1)         # 增加通道维变成 [B, 1, N]
            
        # 在通道维度 (dim=1) 拼接
        points_with_prior = torch.cat([points, probs_A], dim=1) 
        # (注意原代码这里有一行覆写了 points_with_prior，去掉了 probs_A，如果你要用它，请把下面这行注释掉)
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
            
            # 💡 负样本指标计算 (基于高斯真值 < 0.1 且 在管内)
            neg_mask = (targets_wp < 0.1) & combined_mask 
            if neg_mask.sum() > 0:
                neg_probs = probs_B[neg_mask]
                neg_max_probs.append(neg_probs.max().item())
                num_false_pos = (neg_probs > 0.5).float().sum()
                fpr = num_false_pos / neg_probs.numel()
                neg_fpr_list.append(fpr.item())

            # 💡 正样本指标计算 (基于高斯真值 > 0.8 且 在管内)
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

    # ==========================================
    # 💡 计算并打印 Model A 的数据完备率
    # ==========================================
    if total_gt_high_conf > 0:
        recall_rate = (kept_by_model_A / total_gt_high_conf) * 100.0
    else:
        recall_rate = 100.0 # 理论上应该有高置信度点，这里防除零保护

    print(f"[Model B - Ep{epoch_idx}] |\n"
          f"  > Probs: SE:{avg_se_mean:.3f} | Mid:{avg_mid_mean:.3f} | NegMax:{avg_neg_max:.3f} | FPR:{avg_fpr:.4f} | FNR:{avg_mid_fnr:.4f}\n"
          f"  > Graph: Phi_s:{avg_phi_s:.3f} | Phi_c:{avg_phi_c:.3f} | L_str:{avg_l_str:.4f} | L_cost:{avg_l_cost:.4f}\n"
          f"  > Data Completeness: {kept_by_model_A}/{total_gt_high_conf} ({recall_rate:.2f}%) of GT>0.8 passed tube_thresh") 

    return total_loss / max(processed_batches, 1)

def generate_stage_b_offline_dataset(
    model_A, 
    original_loader, # 传入 batch_size=1 的 DataLoader
    save_dir, 
    device, 
    tube_thresh=0.4
):
    """
    基于 Model A 生成 Model B 的专属管内数据集。
    保留了 'points', 'labels', 'waypoints' 的键名，以便复用原有的 Dataset 代码。
    """
    os.makedirs(save_dir, exist_ok=True)
    model_A.eval()
    model_A.to(device)

    print(f"开始生成离线数据集...")
    print(f"保存目录: {save_dir}")
    
    saved_count = 0

    with torch.no_grad():
        for idx, (points, targets, valid_mask, waypoints_list) in enumerate(tqdm(original_loader)):
            # 因为 batch_size=1
            # points: [1, N, D], targets: [1, N, 2], valid_mask: [1, N]
            points = points.to(device)
            valid_mask = valid_mask.to(device)

            # --- 适配你的 Model A 的维度 ---
            # 你的网络通常需要 [B, D, N]，如果最后一维大于等于5，说明当前是 [B, N, D]，需要翻转
            if points.shape[-1] >= 5: 
                points_input = points.permute(0, 2, 1) # 变成 [1, D, N]喂给模型
            else:
                points_input = points

            # --- 运行 Model A 获取概率 ---
            output_A = model_A(points_input, mask=valid_mask)
            logits_A = output_A[0] if isinstance(output_A, (tuple, list)) else output_A
            probs_A = torch.sigmoid(logits_A) 

            # --- 展平所有维度，方便进行 Numpy 级操作 ---
            # 展平为一维向量，避免复杂的维度判断
            probs_A_flat = probs_A.reshape(-1)
            valid_mask_flat = valid_mask.reshape(-1)

            # 计算联合掩码 (管内 + 非 Padding)
            combined_mask_flat = valid_mask_flat & (probs_A_flat > tube_thresh)
            
            # --- 转换到 CPU Numpy ---
            mask_np = combined_mask_flat.cpu().numpy()               # [N]
            pts_np = points.squeeze(0).cpu().numpy()                 # [N, D]  注意我们用原始的 points
            targets_np = targets.squeeze(0).cpu().numpy()            # [N, 2]  保留了完整的两套真值
            probs_np = probs_A_flat.cpu().numpy().reshape(-1, 1)     # [N, 1]  将概率变成特征列

            # --- 💡 核心：利用布尔索引进行过滤 ---
            filtered_pts = pts_np[mask_np]           # 变成 [N', D] 
            filtered_targets = targets_np[mask_np]   # 变成 [N', 2] (两套真值都留着)
            filtered_probs = probs_np[mask_np]       # 变成 [N', 1]

            if filtered_pts.shape[0] == 0:
                print(f"\n[警告] 样本 {idx} 过滤后点数为 0 (管内无有效点)，跳过。")
                continue

            # --- 💡 特征重构：拼加得分特征 ---
            # 原来的特征是 D 维，现在变成 D+1 维
            final_points = np.concatenate([filtered_pts, filtered_probs], axis=1)

            # 获取航路点 [M, 3]
            waypoints_np = waypoints_list[0].cpu().numpy() 

            # --- 💡 按照你原 Dataset 的键名保存 ---
            save_path = os.path.join(save_dir, f"stage_b_data_{saved_count:05d}.npz")
            np.savez_compressed(
                save_path,
                points=final_points,       # [N', D+1] (管内点 + A预测的特征)
                labels=filtered_targets,   # [N', 2]   (两套真值完好无损)
                waypoints=waypoints_np     # [M, 3]    (原封不动)
            )
            saved_count += 1
            
    print(f"数据生成完成！共提取了 {saved_count} 个有效样本。")

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
            points_input = points[:, :4, :]

            # 💡 2. 传入 mask 给模型
            output = model(points_input, mask=mask)
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

            # 负样本统计 (< 0.4)
            neg_mask = (targets_wp < 0.4) & combined_mask 
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

def train_one_epoch_B_offline(model_B, loader, criterion, optimizer, device, epoch_idx):
    """
    航路点网络 (Model B) 的极简训练循环 (适配离线生成的 Stage B 数据集)
    特点: 不再需要 model_A 和 tube_thresh，数据自带管内过滤和先验特征。
    """
    model_B.train()
    
    total_loss = 0.0
    processed_batches = 0
    
    # --- Statistics Lists ---
    max_probs, neg_max_probs, neg_fpr_list = [], [], []
    se_mean_probs, mid_mean_probs, mid_fnr_list = [], [], []
    ep_l_str, ep_l_cost, ep_phi_s, ep_phi_c = [], [], [], []

    for batch_idx, (points, targets, valid_mask, gt_waypoints_list) in enumerate(loader):
        points = points.to(device)
        valid_mask = valid_mask.to(device)
        
        # 💡 依然只取第二通道作为航路点真值 (离线数据保留了两套真值)
        targets_wp = targets[..., 1:2].to(device) 

        # 调整特征维度 [B, N, D+1] -> [B, D+1, N]
        if points.shape[-1] >= 5:
            points = points.permute(0, 2, 1) 
        
        # ==========================================
        # 💡 特征选择 (适配你的消融实验需求)
        # 离线数据中的 points 包含: [原始特征..., Model_A_Prob]
        # 最后一维 points[:, -1:, :] 必定是 Model_A 的预测概率
        # ==========================================
        # 如果你想保留全部特征：
        points_input = points 
        
        # 如果你只想用前4个特征(如 xyz + 某个特征) 且 不要 Model A的概率：
        # points_input = torch.cat([points[:, :3, :]], dim=1)


        # 消融前三个特征和后四个特征
        points_input = torch.cat([points[:, :3, :], points[:, -4:, :]], dim=1)

        # ------------------------------------------
        # 开始训练 Model B (直接前向传播)
        # ------------------------------------------
        optimizer.zero_grad()
        
        # 此时的 valid_mask 就是去除 padding 的掩码，管外点已被离线物理删除
        output_B = model_B(points_input, mask=valid_mask) 
        logits_B = output_B[0] if isinstance(output_B, (tuple, list)) else output_B

        # 💡 Loss 计算
        loss, metrics = criterion(logits_B, targets_wp, points_input, mask=valid_mask)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        processed_batches += 1
        
        ep_l_str.append(metrics.get("loss_straight", 0.0))
        ep_l_cost.append(metrics.get("loss_cost", 0.0))
        ep_phi_s.append(metrics.get("phi_s", 0.0))
        ep_phi_c.append(metrics.get("phi_c", 0.0))

        # ==========================================
        # 💡 统计指标时，全部基于 valid_mask 进行过滤
        # ==========================================
        with torch.no_grad():
            probs_B = torch.sigmoid(logits_B)
            if probs_B.dim() == 3: probs_B = probs_B.squeeze()   # [B, N]
            if targets_wp.dim() == 3: targets_wp = targets_wp.squeeze() # [B, N]

            # 提取管内最大概率
            if valid_mask.sum() > 0:
                max_probs.append(probs_B[valid_mask].max().item())
            
            # 💡 负样本指标计算 (< 0.1)
            neg_mask = (targets_wp < 0.1) & valid_mask 
            if neg_mask.sum() > 0:
                neg_probs = probs_B[neg_mask]
                neg_max_probs.append(neg_probs.max().item())
                fpr = (neg_probs > 0.5).float().sum() / neg_probs.numel()
                neg_fpr_list.append(fpr.item())

            # 💡 正样本指标计算 (> 0.8)
            pos_mask = (targets_wp > 0.8) & valid_mask
            
            # 提取起终点特征 (基于完整的 points)
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
                    fnr = (mid_probs < 0.5).float().sum() / mid_probs.numel()
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


def validate_B_offline(model_B, loader, criterion, device):
    """
    航路点网络 (Model B) 的离线验证循环
    不再需要加载 Model A，极速完成验证
    """
    model_B.eval()
    
    total_loss = 0.0
    processed_batches = 0

    neg_max_probs, neg_fpr_list = [], []
    se_mean_probs, mid_mean_probs, mid_fnr_list = [], [], []

    with torch.no_grad():
        for points, targets, valid_mask, gt_waypoints_list in loader:
            points = points.to(device)
            valid_mask = valid_mask.to(device)
            targets_wp = targets[..., 1:2].to(device) 
            
            if points.shape[-1] >= 5:
                points = points.permute(0, 2, 1)

            # --- 特征选择逻辑同 Train ---
            points_input = points 
            # points_input = torch.cat([points[:, :3, :]], dim=1) # 示例：仅取前四维特征
            points_input = torch.cat([points[:, :3, :], points[:, -4:, :]], dim=1)
            
            # 直接推断
            output_B = model_B(points_input, mask=valid_mask)
            logits_B = output_B[0] if isinstance(output_B, (tuple, list)) else output_B

            # Loss 计算
            loss, _ = criterion(logits_B, targets_wp, points_input, mask=valid_mask)
            total_loss += loss.item()
            processed_batches += 1

            # --- 检测指标统计 ---
            probs_B = torch.sigmoid(logits_B)
            if probs_B.dim() == 3: probs_B = probs_B.squeeze()
            if targets_wp.dim() == 3: targets_wp = targets_wp.squeeze()

            # 负样本统计 (< 0.4)
            neg_mask = (targets_wp < 0.4) & valid_mask 
            if neg_mask.sum() > 0:
                neg_probs = probs_B[neg_mask]
                neg_max_probs.append(neg_probs.max().item())
                fpr = (neg_probs > 0.5).float().sum() / neg_probs.numel()
                neg_fpr_list.append(fpr.item())

            # 正样本统计 (> 0.8)
            pos_mask = (targets_wp > 0.8) & valid_mask
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
                    fnr = (mid_probs < 0.5).float().sum() / mid_probs.numel()
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

    train_loader = build_dataloader(train_files, batch_size=128, shuffle=True)
    val_loader   = build_dataloader(val_files, batch_size=128, shuffle=False)

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
        
        model_A = get_model(num_classes=1, input_dim=4, dropout_p=0.0).to(device)
        
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
        # model_A = get_model(num_classes=1, input_dim=real_input_dim, dropout_p=0.2).to(device)
        # if os.path.exists(MODEL_A_CKPT):
        #     # 1. 先把权重读取到内存字典中
        #     checkpoint = torch.load(MODEL_A_CKPT, map_location=device)
        #     new_state_dict = {}
            
        #     # 2. 遍历字典，翻译 Key 的名字
        #     for k, v in checkpoint.items():
        #         # 替换 Encoder 名称
        #         if 'enc1.' in k: k = k.replace('enc1.', 'enc.0.')
        #         elif 'enc2.' in k: k = k.replace('enc2.', 'enc.1.')
        #         elif 'enc3.' in k: k = k.replace('enc3.', 'enc.2.')
        #         elif 'enc4.' in k: k = k.replace('enc4.', 'enc.3.')
        #         elif 'enc5.' in k: k = k.replace('enc5.', 'enc.4.')
                
        #         # 替换 Decoder 名称
        #         # 注意：我们在 __init__ 中是用 range(4, -1, -1) 倒序生成 dec 的
        #         # 所以原来的 dec5 变成了现在的 dec.0，dec4 变成了 dec.1，以此类推
        #         elif 'dec5.' in k: k = k.replace('dec5.', 'dec.0.')
        #         elif 'dec4.' in k: k = k.replace('dec4.', 'dec.1.')
        #         elif 'dec3.' in k: k = k.replace('dec3.', 'dec.2.')
        #         elif 'dec2.' in k: k = k.replace('dec2.', 'dec.3.')
        #         elif 'dec1.' in k: k = k.replace('dec1.', 'dec.4.')
                
        #         new_state_dict[k] = v
                
        #     # 3. 将翻译好的新字典喂给模型
        #     model_A.load_state_dict(new_state_dict)
        #     print(f"[*] ✅ 成功通过字典映射加载 Model A 管道先验权重!")
            
        # model_A.eval() # 锁定 Dropout 和 BatchNorm
        # for param in model_A.parameters():
        #     param.requires_grad = False # 彻底切断梯度，省下海量显存！
            
        
        # =========================================================
        # 🚀 💡 新增核心：离线数据集生成与重载
        # =========================================================
     
        # offline_train_dir = r"C:\Users\Administrator\Desktop\experiments\train_data16\train"
        # offline_val_dir = r"C:\Users\Administrator\Desktop\experiments\train_data16\val"
        
        # # 检查是否已经生成过（如果文件夹不存在或为空，则触发生成）
        # if not os.path.exists(offline_train_dir) or len(os.listdir(offline_train_dir)) == 0:
        #     print("\n[*] 📦 发现尚未生成 Stage B 的纯净管内数据，开始离线提取...")
        #     # 注意：这里的 train_files 和 val_files 是你原本传入第一阶段的数据列表
        #     # 必须用 batch_size=1 构建临时的 Loader 来精准提取
        #     tmp_train_loader = build_dataloader(train_files, batch_size=1, shuffle=False)
        #     generate_stage_b_offline_dataset(model_A, tmp_train_loader, offline_train_dir, device, tube_thresh=0.4)
            
        #     print("\n[*] 📦 开始提取验证集管内数据...")
        #     tmp_val_loader = build_dataloader(val_files, batch_size=1, shuffle=False)
        #     generate_stage_b_offline_dataset(model_A, tmp_val_loader, offline_val_dir, device, tube_thresh=0.4)
        # else:
        #     print(f"\n[*] 📦 检测到已存在离线数据 ({offline_train_dir})，直接跳过生成步骤！")

        # # 读取新生成的纯净管内数据集
        # offline_train_files = glob.glob(os.path.join(offline_train_dir, "*.npz"))
        # offline_val_files = glob.glob(os.path.join(offline_val_dir, "*.npz"))
        
        # # 使用你原本完美的 collate_fn 重新构建 DataLoader (恢复你想要的 Batch Size，比如 8)
        # # 因为点数变少了，这里 batch_size 甚至可以开得更大！
        # train_loader_B = build_dataloader(offline_train_files, batch_size=32, shuffle=True)
        # val_loader_B = build_dataloader(offline_val_files, batch_size=32, shuffle=False)
        # print(f"[*] ♻️ DataLoader 重构完毕，新的训练集样本数: {len(offline_train_files)}")

        all_files_B = glob.glob(os.path.join(r"C:\Users\Administrator\Desktop\experiments\train_data16\enriched_all_stage_B_Augmented", "*.npz"))
        train_files_B = all_files_B[:int(len(all_files_B)*0.8)]
        val_files_B = all_files_B[int(len(all_files_B)*0.8):]
        train_loader_B = build_dataloader(train_files_B, batch_size=32, shuffle=True)
        val_loader_B = build_dataloader(val_files_B, batch_size=32, shuffle=False)
        # =========================================================

        # 💡 2. 实例化 Model B (核心：输入维度 + 1，因为拼接了 P_tube)
        # 注意：这里你可以把 dropout 调高一点，防止在小规模正样本上过拟合
        # 测试发现share_planes比较大时效果更好，stride保证最终下采样层的点数在32-128区间，nsample貌似没有什么影响
        model_B = get_model(num_classes=1, input_dim=7, dropout_p=0.2, blocks=[1,2,2], stride=[1,2,2], nsample=[8,8,8], share_planes=16).to(device)
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
            # train_loss = train_one_epoch_B(
            #     model_B=model_B, 
            #     model_A=model_A, 
            #     loader=train_loader, 
            #     criterion=criterion_B, 
            #     optimizer=optimizer_B, 
            #     device=device, 
            #     epoch_idx=epoch, 
            #     tube_thresh=0.4 # 宽容截流阈值
            # )
            
            # # 💡 调用专属的 validate_B
            # val_loss = validate_B(
            #     model_B=model_B, 
            #     model_A=model_A, 
            #     loader=val_loader, 
            #     criterion=criterion_B, 
            #     device=device,
            #     tube_thresh=0.4
            # )

            # 💡 调用专属的离线训练函数，不再传入 model_A 和 tube_thresh
            train_loss = train_one_epoch_B_offline(
                model_B=model_B, 
                loader=train_loader_B,  # <-- 使用读取离线纯净数据的新 loader
                criterion=criterion_B, 
                optimizer=optimizer_B, 
                device=device, 
                epoch_idx=epoch
            )
            
            # 💡 调用专属的离线验证函数，同样不再传入 model_A 和 tube_thresh
            val_loss = validate_B_offline(
                model_B=model_B, 
                loader=val_loader_B,    # <-- 使用读取离线纯净验证数据的新 loader
                criterion=criterion_B, 
                device=device
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
