import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# 导入你的模块
from src.data.data_loader import PathPointDataset, collate_fn
from src.models.pointnet_transfomer2.my_model import get_model
from torch.utils.data import DataLoader

def visualize_result(xyz, target, pred_prob, threshold=0.5):
    """
    3D 可视化函数 - 改进版：左右图采用相同的绘制风格
    """
    xyz = xyz.cpu().numpy()
    target = target.cpu().numpy()
    pred_prob = pred_prob.cpu().numpy()

    # 1. 提取需要高亮显示的点
    pred_mask = pred_prob > threshold
    pred_points = xyz[pred_mask]
    pred_colors = pred_prob[pred_mask]

    gt_mask = target > 0.5
    gt_points = xyz[gt_mask]
    gt_colors = target[gt_mask]

    fig = plt.figure(figsize=(16, 8))

    # --- 定义统一的绘图参数 ---
    scatter_kwargs = {
        'cmap': 'jet',    # 统一使用 jet 颜色映射
        's': 20,          
        'vmin': 0.0,      
        'vmax': 1.0,      
        'alpha': 1.0,     
        'edgecolor': 'none' 
    }

    # --- 左图：Ground Truth ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Ground Truth Path (Target=1.0, solid Red)")
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='gray', s=1, alpha=0.05)

    p1 = None
    if len(gt_points) > 0:
        p1 = ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                         c=gt_colors, label='GT', **scatter_kwargs)

    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.view_init(elev=30, azim=-60)

    # --- 右图：Prediction ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Prediction (Conf > {threshold})")
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='gray', s=1, alpha=0.05)

    p2 = None
    if len(pred_points) > 0:
        p2 = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                         c=pred_colors, label='Pred', **scatter_kwargs)
    
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.view_init(elev=30, azim=-60)

    # --- 添加统一的 Colorbar ---
    plot_handle = p2 if p2 is not None else p1
    
    # 【防御性修复】万一模型全预测错(无预测点)且GT也无点，避免报错
    if plot_handle is None:
        plot_handle = ax2.scatter([xyz[0,0]], [xyz[0,1]], [xyz[0,2]], c=[0.0], s=0, **scatter_kwargs)

    cbar = fig.colorbar(plot_handle, ax=[ax1, ax2], shrink=0.7, location='right', pad=0.02)
    cbar.set_label('Probability / Confidence Score')

    plt.show()

@torch.no_grad()
def evaluate(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    model = get_model(num_classes=1, input_dim=6).to(device)

    print(f"Loading checkpoint from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: 权重文件不存在 -> {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. 加载数据
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if len(all_files) == 0:
        print(f"Error: 在该目录下找不到 .npz 数据 -> {data_dir}")
        return

    print(f"Found {len(all_files)} samples. Starting evaluation...")

    dataset = PathPointDataset(all_files)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 3. 推理循环
    for i, (points, targets) in enumerate(test_loader):
        points = points.to(device)   # [B, N, 6]
        targets = targets.to(device) # [B, N, 1]

        # 【核心修复 1】维度调整并保证内存连续性
        points_trans = points.permute(0, 2, 1).contiguous() # [B, 6, N]

        # 推理
        output = model(points_trans)
        
        # 【核心修复 2】兼容模型可能返回的 Tuple 结构
        logits = output[0] if isinstance(output, (tuple, list)) else output
        
        probs = torch.sigmoid(logits)  # [B, N, 1]

        # 取出第一个样本进行可视化
        xyz_vis = points[0, :, :3]      # [N, 3] 
        target_vis = targets[0, :, 0]   # [N]
        prob_vis = probs[0, :, 0]       # [N]

        # 打印统计信息
        max_conf = prob_vis.max().item()
        path_conf = prob_vis[target_vis==1].mean().item() if (target_vis==1).sum() > 0 else 0

        print(f"\n--- Sample {i} ({os.path.basename(all_files[i])}) ---")
        print(f"Max Predicted Confidence in scene: {max_conf:.4f}")
        print(f"Avg Confidence on GT Path points: {path_conf:.4f}")

        # 可视化
        visualize_result(xyz_vis, target_vis, prob_vis, threshold=0.5)

        cmd = input("Press Enter for next sample, or 'n' to stop: ")
        if cmd.lower() == 'n':
            break

if __name__ == "__main__":

    # 1. 数据文件夹路径
    DATA_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\src\data\data_for_train\train_data"
    
    # 2. 权重文件路径 (指向 experiments 下的最佳模型)
    CKPT_PATH = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\experiments\checkpoints\best_model.pth"

    print("-" * 50)
    print(f"Data Dir: {DATA_DIR}")
    print(f"Model Path: {CKPT_PATH}")
    print("-" * 50)

    evaluate(CKPT_PATH, DATA_DIR)