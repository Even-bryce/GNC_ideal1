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
SAVE_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\experiments\checkpoints"

# 2. 真实的训练数据路径
DATA_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\src\data\data_for_train\train_data4"

GAMMA = 2 
TOTAL_EPOCHS = 50         
# ---------------------      
# ---------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    # --- Warm-up 策略 ---
    if epoch_idx <= 50:
        criterion.w_straight = 0.0
        criterion.delta_s = 0.00
        criterion.delta_d = 0.00  
        phase_name = "Warm-up (FOCAL Only)"
    else:
        criterion.w_straight = 2.0
        criterion.delta_s = 0.8
        criterion.delta_d = 0.15    
        phase_name = "Refinement (Geo Loss Active)"

    model.train()
    total_loss = 0.0
    
    # --- Statistics Lists ---
    max_probs, neg_max_probs, neg_fpr_list = [], [], []
    se_mean_probs, mid_mean_probs = [], []
    mid_fnr_list = []  # 💡 新增：专门记录中间航路点的漏检率

    for batch_idx, (points, targets, gt_waypoints_list) in enumerate(loader):
        points = points.to(device)
        targets = targets.to(device)

        # 确保 points 是 [B, C, N] 格式
        if points.shape[-1] >= 6:
            points = points.permute(0, 2, 1) 
        
        # 提取物理坐标用于 Loss: [B, N, 3]
        xyz = points[:, :3, :].permute(0, 2, 1).contiguous()

        optimizer.zero_grad()
        
        # 1. 模型前向传播 (桥接模式会自动处理 pxo 格式)
        output = model(points) 
        
        # 2. 解包 logits
        logits = output[0] if isinstance(output, (tuple, list)) else output

        # 3. 计算复合 Loss
        loss = criterion(logits, targets, xyz)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # --- Monitoring Logic ---
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            
            if probs.dim() == 3: probs = probs.squeeze(-1)   # [B, N]
            if targets.dim() == 3: targets = targets.squeeze(-1) # [B, N]

            max_probs.append(probs.max().item())
            
            # --- 负样本统计 (误报监控) ---
            neg_mask = (targets < 0.1) 
            if neg_mask.sum() > 0:
                neg_probs = probs[neg_mask]
                neg_max_probs.append(neg_probs.max().item())
                
                # 误报：真值不是航路点，但预测概率 > 0.5
                num_false_pos = (neg_probs > 0.5).float().sum()
                fpr = num_false_pos / neg_probs.numel()
                neg_fpr_list.append(fpr.item())

            # --- 正样本统计 (漏检监控) ---
            pos_mask = (targets > 0.9)
            
            # 自动提取距离特征 (适配起点和终点)
            if points.shape[1] >= 6:  # 只要总维度够，直接取倒数两列
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
                    
                    # 💡 新增：漏检（真值是航路点，但预测概率 < 0.5）
                    num_false_neg = (mid_probs < 0.5).float().sum()
                    fnr = num_false_neg / mid_probs.numel()
                    mid_fnr_list.append(fnr.item())

    # --- Averages ---
    avg_neg_max = np.mean(neg_max_probs) if len(neg_max_probs) > 0 else 0.0
    avg_fpr = np.mean(neg_fpr_list) if len(neg_fpr_list) > 0 else 0.0
    avg_se_mean = np.mean(se_mean_probs) if len(se_mean_probs) > 0 else 0.0
    avg_mid_mean = np.mean(mid_mean_probs) if len(mid_mean_probs) > 0 else 0.0
    avg_mid_fnr = np.mean(mid_fnr_list) if len(mid_fnr_list) > 0 else 0.0  # 💡 新增平均漏检率

    # 💡 更新打印面板，加入 FNR(漏检)
    print(f"[Debug Ep{epoch_idx}] {phase_name} | "
          f"SE:{avg_se_mean:.3f} | "    
          f"Mid:{avg_mid_mean:.3f} | "  
          f"NegMax:{avg_neg_max:.3f} | "
          f"FPR(误报):{avg_fpr:.4f} | "
          f"FNR(漏检):{avg_mid_fnr:.4f}")         

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    【修复】验证集应使用与训练集一致的 criterion，
    否则绘制的 Loss 曲线将因为量级差异失去对比意义。
    """
    model.eval()
    total_loss = 0.0

    # 验证集不需要动态改变直线性权重，可根据需求固定
    # criterion.w_straight = 5.0 

    for points, targets, gt_waypoints_list in loader:
        points = points.to(device)
        targets = targets.to(device)
        
        if points.shape[-1] >= 6:
            points = points.permute(0, 2, 1)
            
        xyz = points[:, :3, :].permute(0, 2, 1).contiguous()

        output = model(points)
        logits = output[0] if isinstance(output, (tuple, list)) else output

        # 使用同样的复合 Loss 计算验证误差
        loss = criterion(logits, targets, xyz)
        total_loss += loss.item()

    return total_loss / len(loader)

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
    
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")

    # ==========================================
    # 2. Data Loading (整体挪到了模型前面)
    # ==========================================
    data_dir = DATA_DIR
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if len(all_files) == 0:
        print(f"Error: No .npz files found in {data_dir}")
        return

    split = int(len(all_files) * 0.8)
    train_files = all_files[:split]
    val_files = all_files[split:]

    train_loader = build_dataloader(train_files, batch_size=32, shuffle=True)
    val_loader   = build_dataloader(val_files, batch_size=32, shuffle=False)

    # --- 【最简单的动态探针】 ---
    # 拿一个 batch 出来探测真实维度 (DataLoader 输出的 points 形状通常为 [B, N, C])
    sample_points, _, curr_gt_waypoints= next(iter(train_loader))
    real_input_dim = sample_points.shape[-1]
    print(f"[*] 动态检测到数据特征维度为: {real_input_dim}")

    # ==========================================
    # 3. Model (传入刚刚探测出来的真实维度)
    # ==========================================
    model = get_model(num_classes=1, input_dim=real_input_dim).to(device)

    # 4. Loss Configuration
    criterion = get_loss(
        w_bce=20,
        w_straight=0,
        w_safety=0,
        w_conn=0.0,
        gamma=GAMMA,
        delta_s=0.5,
        r_corridor=0.03,
        r_local=0.05,
        rho=25600.0,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')

    print(f"Start training on {device} with gamma={GAMMA}...")
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate (传入 criterion 保证评估尺度一致)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(
            f"[Epoch {epoch:03d}/{TOTAL_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        # --- Save Strategy ---
        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  >>> New Best Model Saved! (Val Loss: {val_loss:.4f})")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth"))
            plot_loss_curve(train_loss_list, val_loss_list, save_dir)

    print("绘制最终 Loss 曲线...")
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