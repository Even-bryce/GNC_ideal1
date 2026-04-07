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
DATA_DIR = r"C:\Users\Administrator\Desktop\experiments\train_data6"

START_EPOCH = 41  # 如果从头训练填 1；如果调参直接从第 41 轮开始，填 41
PRETRAINED_CKPT = r"C:\Users\Administrator\Desktop\experiments\checkpoints\ckpt_epoch_40.pth" # 填入你第40轮保存的权重路径

GAMMA = 2
ALPHA = 0.6
TOTAL_EPOCHS = 100         
# ---------------------      
# ---------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    # --- Warm-up 策略 ---
    warmup_epochs = 40
    if epoch_idx <= warmup_epochs:
        criterion.w_bce = 20
        criterion.w_straight = 0.0
        criterion.w_cost = 0.0
        criterion.w_safety = 0.1  
        phase_name = "Warm-up (FOCAL Only)"
    else:
        criterion.w_bce = 20
        criterion.w_straight = 5
        criterion.w_cost = 4
        criterion.w_safety = 0.2
        criterion.delta_s = 0.8
        criterion.delta_d = 0.2
        criterion.tau_s = 0.4
        criterion.tau_c = 0.3
        phase_name = "Refinement (other Loss Active)"

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
        loss = criterion(logits, targets, points)
        
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
            pos_mask = (targets > 0.8)
            
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
        loss = criterion(logits, targets, points)
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
    # 2. Data Loading
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
    sample_points, _, curr_gt_waypoints = next(iter(train_loader))
    real_input_dim = sample_points.shape[-1]
    print(f"[*] 动态检测到数据特征维度为: {real_input_dim}")

    # ==========================================
    # 3. Model
    # ==========================================
    model = get_model(num_classes=1, input_dim=real_input_dim, dropout_p=0).to(device)

    # 💡 新增：动态加载预训练权重
    if START_EPOCH > 1:
        if os.path.exists(PRETRAINED_CKPT):
            print(f"\n[*] ⚡ 极速调参模式开启！正在加载第 {START_EPOCH-1} 轮预训练权重...")
            model.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=device))
            print(f"[*] ✅ 权重加载成功: {PRETRAINED_CKPT}\n")
        else:
            print(f"\n[!] ❌ 找不到预训练权重文件: {PRETRAINED_CKPT}")
            print("[!] 请检查路径是否正确！程序退出。")
            return

    # 4. Loss Configuration
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
        r_corridor=0.03,
        r_local=0.05,
        rho=25600.0,
        # 💡 --- 新增的点对筛选与奖惩参数 ---
        R_nms=0.15,       # 骨架点 NMS 抑制半径
        K_local=3,
        K_pairs=3,       # 局部允许的候选点最大数量 (增加多样性)
        alpha2=2.0,      # 直线 Loss 的指数敏感度
        alpha3=1.0,      # 成本 Loss 的指数敏感度
        tau_s=0,       # 直线通畅度及格线 (大于奖励，小于惩罚)
        tau_c=0,       # 成本(步长)及格线
        
    ).to(device)


    initial_lr = 1e-4 if START_EPOCH == 1 else 1e-5

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=5e-5
    )

    step_size = 15 if START_EPOCH == 1 else 10 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    

    train_loss_list = []
    val_loss_list = []

    # 💡 新增：实例化早停对象 (容忍 15 个 epoch 没有显著提升)
    early_stopping = EarlyStopping(patience=15, delta=0.001, save_dir=save_dir)

    print(f"Start training on {device} with gamma={GAMMA}...")
    
    for epoch in range(START_EPOCH, TOTAL_EPOCHS + 1):

        # # ==========================================
        # # 🚀 终极杀招：在 Refinement 阶段冻结主干网络
        # # ==========================================
        # warmup_epochs = 40  
        
        # if epoch == warmup_epochs + 1:
        #     print(f"\n[{'='*40}]")
        #     print("🚀 进入 Refinement 阶段！执行主干网络冻结 (Backbone Freezing)！")
            
        #     # 💡 1. 更新白名单：包含新设计的 MoE 头的四个核心组件
        #     moe_head_keywords = ["head_base", "expert_recall", "expert_refine", "gate"]
            
        #     for name, param in model.named_parameters():
        #         # 如果当前层的名字里，不包含上面任何一个关键字，就冻结它
        #         if not any(keyword in name for keyword in moe_head_keywords):
        #             param.requires_grad = False
        #         else:
        #             print(f"  ✅ 保持活动状态 (接受微调): {name}")
            
        #     # 2. 重新配置优化器，只把还活着的参数（分类头）喂给 Adam
        #     active_params = filter(lambda p: p.requires_grad, model.parameters())
            
        #     # 💡 给新的 MoE 头一点活力，学习率设为 5e-5，同时加上一点 weight_decay 防过拟合
        #     optimizer = torch.optim.Adam(active_params, lr=5e-5, weight_decay=1e-3)
            
        #     # 3. 重新配置学习率衰减器
        #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
            
        #     print("🚀 优化器已重置，跷跷板已被彻底焊死！双分支专家头开始接管比赛！")
        #     print(f"[{'='*40}]\n")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate 
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
        # 无论如何保存最新的 epoch
        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

        # 💡 新增：调用早停判定器 (它内部会自动判断并保存 best_model.pth)
        early_stopping(val_loss, model)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth"))
            plot_loss_curve(train_loss_list, val_loss_list, save_dir)

        # 💡 新增：触发早停的退出逻辑
        if early_stopping.early_stop:
            print(f"🛑 触发早停机制！模型在最近 {early_stopping.patience} 个 Epoch 内没有提升，提前结束训练。")
            break

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