import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 导入你的模块
from data_loader import build_dataloader
from my_model import get_model

def visualize_result(xyz, target, pred_prob, threshold=0.5):
    """
    xyz: [N, 3]
    target: [N] (0 or 1)
    pred_prob: [N] (0.0 ~ 1.0)
    """
    xyz = xyz.cpu().numpy()
    target = target.cpu().numpy()
    pred_prob = pred_prob.cpu().numpy()

    # 1. 提取不同类别的点
    # 预测为路径的点 (Prob > threshold)
    pred_mask = pred_prob > threshold
    pred_points = xyz[pred_mask]
    
    # 真实标签为路径的点 (Ground Truth)
    gt_mask = target > 0.5
    gt_points = xyz[gt_mask]

    # 背景点 (为了绘图清晰，通常只画一部分背景或者不画)
    # bg_points = xyz[~gt_mask & ~pred_mask]

    fig = plt.figure(figsize=(15, 6))

    # --- 左图：Ground Truth ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Ground Truth Path")
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgray', s=1, alpha=0.1, label='Map')
    ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='green', s=20, label='GT Path')
    ax1.legend()

    # --- 右图：Model Prediction ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Prediction (Conf > {threshold})")
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgray', s=1, alpha=0.1)
    
    # 用颜色深浅表示置信度
    if len(pred_points) > 0:
        p = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                        c=pred_prob[pred_mask], cmap='jet', s=20, vmin=0, vmax=1, label='Pred')
        fig.colorbar(p, ax=ax2)
    
    plt.show()

@torch.no_grad()
def evaluate(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    # 注意：确保这里的 input_dim 和你训练时一致 (6)
    model = get_model(num_classes=1, input_dim=6).to(device)
    
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. 加载数据 (使用 'val' 或 'test' 模式)
    # 注意：这里我们 batch_size=1，方便一张张图看
    test_loader = build_dataloader(
        data_files=[os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')],
        batch_size=1, 
        shuffle=False,
        num_workers=0
    )

    print("Start Evaluation... (Close the plot window to see the next sample)")

    for i, (points, targets) in enumerate(test_loader):
        """
        points: [B, N, 6]
        targets: [B, N, 1]
        """
        points = points.to(device)
        targets = targets.to(device)

        # 维度调整适应 PointNet++ 输入 [B, 6, N]
        points_trans = points.permute(0, 2, 1)

        # 推理
        logits = model(points_trans)   # [B, N, 1]
        probs = torch.sigmoid(logits)  # 转为 0~1 概率

        # 取出第一个样本进行可视化
        # points[0, :, :3] 取出 xyz 坐标 (假设前3列是xyz)
        # 你的 points 是 [B, N, 6]，所以 xyz 是 points[0, :, :3]
        xyz_vis = points[0, :, :3]
        target_vis = targets[0, :, 0]
        prob_vis = probs[0, :, 0]

        print(f"\n--- Sample {i} ---")
        print(f"Max Prob: {prob_vis.max().item():.4f}")
        print(f"Mean Prob on Path: {prob_vis[target_vis==1].mean().item():.4f}")
        
        # 可视化
        visualize_result(xyz_vis, target_vis, prob_vis, threshold=0.5)

        # 看了 5 张之后询问是否继续，防止太烦
        if i > 0 and i % 5 == 0:
            cmd = input("Continue? (y/n): ")
            if cmd.lower() == 'n':
                break

if __name__ == "__main__":
    # 修改为你的路径
    CKPT_PATH = "ckpt_epoch_100.pth"  # 训练生成的权重文件
    DATA_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\raw_model\train_data"
    
    evaluate(CKPT_PATH, DATA_DIR)