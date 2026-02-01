import torch
import torch.nn.functional as F
from data_loader import build_dataloader
from my_model import get_model, get_loss, focal_loss
import os
import time
import datetime
import glob
import numpy as np
import matplotlib.pyplot as plt  # <--- 修改 1: 导入绘图库

# --- Configuration ---
# Your absolute path to raw_model
ROOT_PATH = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\raw_model"
GAMMA = 2 
TOTAL_EPOCHS = 20        
# ---------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    # --- Warm-up 策略 ---
    if epoch_idx <= 15:
        criterion.w_straight = 0.0
        criterion.delta_s = 0.01
        criterion.delta_d = 0.06  
        phase_name = "Warm-up (BCE Only)"
    else:
        criterion.w_straight = 5
        criterion.delta_s = 0.6
        criterion.delta_d = 0.06    
        phase_name = "Refinement (Geo Loss Active)"

    model.train()
    total_loss = 0.0
    
    # --- Statistics Lists ---
    max_probs = []
    neg_max_probs = []  
    neg_fpr_list = []   # <--- 保留：你的 FPR 列表

    # [新增] 拆分后的统计列表
    se_mean_probs = []   # 起终点得分
    mid_mean_probs = []  # 中间点得分

    for batch_idx, (points, targets) in enumerate(loader):
        points = points.to(device)
        targets = targets.to(device)

        # points: [B, N, C] -> [B, C, N]
        points = points.permute(0, 2, 1) 
        xyz = points[:, :3, :].permute(0, 2, 1) # [B, N, 3]

        optimizer.zero_grad()
        logits = model(points)
        loss = criterion(logits, targets, xyz)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # --- Monitoring Logic (融合修复版) ---
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            
            # [关键修复 1] 确保维度统一，去掉最后的 1
            if probs.dim() == 3: probs = probs.squeeze(-1)   # [B, N]
            if targets.dim() == 3: targets = targets.squeeze(-1) # [B, N]

            max_probs.append(probs.max().item())
            
            # --- 1. 负样本统计 (FPR & Neg_Max) ---
            # 你的原始逻辑：小于 0.1 或 等于 0 视为负样本
            neg_mask = (targets < 0.1) 
            if neg_mask.sum() > 0:
                neg_probs = probs[neg_mask]
                neg_max_probs.append(neg_probs.max().item())
                
                # 计算 FPR (假阳性率): 负样本中得分 > 0.5 的比例
                num_false_pos = (neg_probs > 0.5).float().sum()
                fpr = num_false_pos / neg_probs.numel()
                neg_fpr_list.append(fpr.item())

            # --- 2. 正样本拆分统计 (Mid_Mean) ---
            pos_mask = (targets > 0.9) # 正样本
            
            # # [关键修复 2] 自动识别维度提取距离通道，防止报错
            # if points.shape[2] == probs.shape[1]: 
            #     # Case A: [B, C, N] (N 在最后) -> 取第4,5通道
            #     dist_start = points[:, 4, :]
            #     dist_end   = points[:, 5, :]
            # else:
            #     # Case B: [B, N, C] (N 在中间) -> 取第4,5特征
            #     dist_start = points[:, :, 4]
            #     dist_end   = points[:, :, 5]


            # [关键修复 2] 自动识别维度提取距离通道，防止报错
            if points.shape[2] == probs.shape[1]: 
                # Case A: [B, C, N] (N 在最后) -> 取第4,5通道
                dist_start = points[:, 6, :]
                dist_end   = points[:, 7, :]
            else:
                # Case B: [B, N, C] (N 在中间) -> 取第4,5特征
                dist_start = points[:, :, 6]
                dist_end   = points[:, :, 7]

            is_start_end = (dist_start < 0.05) | (dist_end < 0.05)

            # 拆分
            mask_SE = pos_mask & is_start_end       # 起终点
            mask_MID = pos_mask & (~is_start_end)   # 中间航路点

            if mask_SE.sum() > 0:
                se_mean_probs.append(probs[mask_SE].mean().item())
            
            if mask_MID.sum() > 0:
                mid_mean_probs.append(probs[mask_MID].mean().item())

    # --- Calculate Epoch Averages ---
    avg_max_prob = np.mean(max_probs)
    avg_neg_max = np.mean(neg_max_probs) if len(neg_max_probs) > 0 else 0.0
    avg_fpr = np.mean(neg_fpr_list) if len(neg_fpr_list) > 0 else 0.0 # <--- 你的 FPR 均值
    
    avg_se_mean = np.mean(se_mean_probs) if len(se_mean_probs) > 0 else 0.0
    avg_mid_mean = np.mean(mid_mean_probs) if len(mid_mean_probs) > 0 else 0.0

    # --- 打印所有指标 (包含 FPR) ---
    print(f"[Debug Ep{epoch_idx}] {phase_name} | "
          f"SE:{avg_se_mean:.3f} | "    # 起终点
          f"Mid:{avg_mid_mean:.3f} | "  # 中间点 (重点关注!)
          f"NegMax:{avg_neg_max:.3f} | "
          f"FPR:{avg_fpr:.4f}")         # <--- 恢复 FPR 打印

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device, gamma):

    model.eval()

    total_loss = 0.0



    for points, targets in loader:

        points = points.to(device)

        targets = targets.to(device)

        points = points.permute(0, 2, 1)

        logits = model(points)

        loss = focal_loss(logits, targets, gamma=gamma)

        total_loss += loss.item()



    return total_loss / len(loader)

    return total_loss / len(loader)

def plot_loss_curve(train_loss, val_loss, save_dir):
    """
    绘制并保存 Loss 曲线
    """
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
    
    # 保存原始数据，防止以后想重画
    # np.savez(os.path.join(save_dir, "loss_data.npz"), 
    #          train_loss=np.array(train_loss), 
    #          val_loss=np.array(val_loss))
    # print(f"Loss曲线已保存至: {os.path.join(save_dir, 'loss_curve.png')}")

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = os.path.join(ROOT_PATH, "checkpoints2")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")

    # 2. Model
    model = get_model(num_classes=1, input_dim=8).to(device)

    # 3. Loss Configuration
    criterion = get_loss(
        w_bce=20,
        w_straight=3,
        w_safety=0,
        w_conn=0.0,
        gamma=GAMMA,
        delta_s=0.5,
        r_corridor=0.03,
        r_local=0.05,
        rho=25600.0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # 4. Data Loading
    data_dir = os.path.join(ROOT_PATH, "train_data2")
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if len(all_files) == 0:
        print(f"Error: No .npz files found in {data_dir}")
        return

    split = int(len(all_files) * 0.8)
    train_files = all_files[:split]
    val_files = all_files[split:]

    train_loader = build_dataloader(train_files, batch_size=8, shuffle=True)
    val_loader   = build_dataloader(val_files, batch_size=8, shuffle=False)

    # --- 修改 2: 初始化 Loss 记录列表 ---
    train_loss_list = []
    val_loss_list = []

    # 5. Training Loop
    best_val_loss = float('inf')

    print(f"Start training on {device} with gamma={GAMMA}...")
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss = validate(model, val_loader, device, gamma=GAMMA)
        
        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- 修改 3: 记录数据 ---
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        # --- Save Strategy ---
        # Save Last
        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  >>> New Best Model Saved! (Val Loss: {val_loss:.4f})")

        # Periodic Save
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth"))
            
            # (可选) 每10个epoch也可以更新一下图，防止训练中途断了没图看
            plot_loss_curve(train_loss_list, val_loss_list, save_dir)

    # --- 修改 4: 训练结束后绘制最终曲线 ---
    print("绘制 Loss 曲线...")
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