import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# 导入你的模块
from data_loader import build_dataloader  # 确保这里不需要改，或者像下面直接用 Dataset
from data_loader import PathPointDataset, collate_fn # 如果 build_dataloader 不好用，我们直接用 Dataset
from my_model import get_model
from torch.utils.data import DataLoader

def visualize_result(xyz, target, pred_prob, threshold=0.5):
    """
    3D 可视化函数
    """
    xyz = xyz.cpu().numpy()
    target = target.cpu().numpy()
    pred_prob = pred_prob.cpu().numpy()

    # 1. 提取不同类别的点
    pred_mask = pred_prob > threshold
    pred_points = xyz[pred_mask]
    
    gt_mask = target > 0.5
    gt_points = xyz[gt_mask]

    fig = plt.figure(figsize=(15, 8))

    # --- 左图：Ground Truth (真实路径) ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Ground Truth Path")
    # 画背景点 (灰色，透明度高)
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgray', s=1, alpha=0.1)
    # 画真实路径点 (绿色)
    if len(gt_points) > 0:
        ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='green', s=10, label='GT')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # --- 右图：Prediction (模型预测) ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Prediction (Conf > {threshold})")
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgray', s=1, alpha=0.1)
    
    # 用颜色深浅表示置信度
    if len(pred_points) > 0:
        p = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                        c=pred_prob[pred_mask], cmap='jet', s=10, vmin=0, vmax=1, label='Pred')
        fig.colorbar(p, ax=ax2, shrink=0.5)
    
    plt.show()

@torch.no_grad()
def evaluate(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    # 注意：确保这里的 input_dim 和你训练时一致 (6)
    model = get_model(num_classes=1, input_dim=6).to(device)
    
    print(f"Loading checkpoint from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: 权重文件不存在 -> {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. 加载数据
    # 直接查找该目录下的所有 npz 文件
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if len(all_files) == 0:
        print(f"Error: 在该目录下找不到 .npz 数据 -> {data_dir}")
        return
    
    print(f"Found {len(all_files)} samples. Starting evaluation...")
    
    # 构造 DataLoader (batch_size=1 方便逐个可视化)
    dataset = PathPointDataset(all_files)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 3. 推理循环
    for i, (points, targets) in enumerate(test_loader):
        points = points.to(device)   # [B, N, 6]
        targets = targets.to(device) # [B, N, 1]

        # 维度调整适应 PointNet++ 输入: [B, N, 6] -> [B, 6, N]
        points_trans = points.permute(0, 2, 1)

        # 推理
        logits = model(points_trans)   # [B, N, 1]
        probs = torch.sigmoid(logits)  # 转为 0~1 概率

        # 取出第一个样本进行可视化
        xyz_vis = points[0, :, :3]      # [N, 3] 取前3维坐标
        target_vis = targets[0, :, 0]   # [N]
        prob_vis = probs[0, :, 0]       # [N]

        # 打印一些统计信息
        max_conf = prob_vis.max().item()
        path_conf = prob_vis[target_vis==1].mean().item() if (target_vis==1).sum() > 0 else 0
        
        print(f"\n--- Sample {i} ---")
        print(f"Max Predicted Confidence: {max_conf:.4f}")
        print(f"Avg Confidence on GT Path: {path_conf:.4f}")
        
        # 可视化
        visualize_result(xyz_vis, target_vis, prob_vis, threshold=0.5)

        # 每看一张图暂停一下，输入 n 退出
        cmd = input("Press Enter for next sample, or 'n' to stop: ")
        if cmd.lower() == 'n':
            break

if __name__ == "__main__":
    # 1. 设置数据路径 (保持你的绝对路径)
    DATA_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\raw_model\train_data"

    # 2. 设置权重文件路径
    # 根据你的截图，checkpoints 文件夹在 global_waypoint_generator 的上一级目录
    # 我们先尝试自动定位到那个文件夹
    
    # 想要评估的文件名 (建议用 best_model.pth，或者你可以改成 ckpt_epoch_100.pth)
    CKPT_NAME = "best_model.pth" 
    # CKPT_NAME = "ckpt_epoch_100.pth" 

    # 尝试构建路径：从 DATA_DIR 往上退3层找到 idea1_code，然后找 checkpoints
    # 逻辑：train_data -> raw_model -> global_waypoint_generator -> idea1_code -> checkpoints
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(DATA_DIR)))
        CKPT_PATH = os.path.join(project_root, "checkpoints", CKPT_NAME)
    except:
        # 如果自动推导失败，就用默认的同级假设
        CKPT_PATH = os.path.join("checkpoints", CKPT_NAME)

    # 打印路径确认
    print("-" * 50)
    print(f"Data Dir: {DATA_DIR}")
    print(f"Model Path: {CKPT_PATH}")
    print("-" * 50)

    # 如果自动推导的路径不对，请手动把下面这行注释打开并填入绝对路径：
    CKPT_PATH = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\raw_model\checkpoints\last_model.pth"

    evaluate(CKPT_PATH, DATA_DIR)