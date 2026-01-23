import torch
from data_loader import build_dataloader
from my_model import get_model, get_loss, weighted_bce_loss
import os
import time
import datetime


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for points, targets in loader:
        """
        points : [B, N, 6]
        targets: [B, N, 1]
        """
        points = points.to(device)
        targets = targets.to(device)

        # [B, 6, N]
        points = points.permute(0, 2, 1)

        # xyz: [B, N, 3] —— 已归一化
        xyz = points[:, :3, :].permute(0, 2, 1)

        optimizer.zero_grad()

        logits = model(points)  # [B, N, 1]

        # 🔥 关键修改：直接把 xyz 传给 loss
        loss = criterion(logits, targets, xyz)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    """
    验证阶段：只看 BCE（不加几何正则）
    """
    model.eval()
    total_loss = 0.0

    for points, targets in loader:
        points = points.to(device)
        targets = targets.to(device)

        points = points.permute(0, 2, 1)
        logits = model(points)

        loss = weighted_bce_loss(logits, targets)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    # 1. 基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "checkpoints"  # 专门建个文件夹存权重，比较整洁
    os.makedirs(save_dir, exist_ok=True)

    # 2. 模型与损失
    model = get_model(num_classes=1, input_dim=6).to(device)

    criterion = get_loss(
        w_bce=1.0,
        w_straight=0.1,
        w_safety=0.1,
        w_conn=0.0,
        delta_s=0.5,
        r_corridor=0.03,
        r_local=0.05,
        rho=30000
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    # 学习率调整策略 (可选，加上效果更好)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 3. 数据加载 (注意这里传入文件列表的逻辑，需适配你的 data_loader)
    # 假设 build_dataloader 内部已经处理好了 glob
    import glob
    data_dir = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\raw_model\train_data"
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    # 简单切分
    split = int(len(all_files) * 0.8)
    train_files = all_files[:split]
    val_files = all_files[split:]

    train_loader = build_dataloader(train_files, batch_size=8, shuffle=True)
    val_loader   = build_dataloader(val_files, batch_size=8, shuffle=False)

    # 4. 训练循环
    best_val_loss = float('inf')

    print(f"Start training on {device}...")
    
    for epoch in range(1, 101):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss = validate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        # --- 保存策略 ---
        
        # 1. 保存每个 epoch 的结果（或每隔几个）作为最新检查点
        # 这样如果断了，你可以加载这个继续训
        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

        # 2. 保存历史最佳 (Best Model)
        # 如果当前验证集 Loss 比历史最低还低，就存一份 best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  >>> New Best Model Saved! (Val Loss: {val_loss:.4f})")

        # 3. 定期归档 (例如每 10 个 epoch 存一个留底)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth"))

if __name__ == "__main__":
    print("开始计时...")
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 将秒数转换为 时:分:秒 格式
    time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f"\n{'='*40}")
    print(f"训练全部结束！总耗时: {time_str}")
    print(f"{'='*40}")
