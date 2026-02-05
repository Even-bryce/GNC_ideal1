import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# 导入你的模块
# 确保这里的路径和你项目结构一致
from data_loader import PathPointDataset, collate_fn
from my_model import get_model
from torch.utils.data import DataLoader

def visualize_result(xyz, target, pred_prob, threshold=0.5):
    """
    3D 可视化函数 - 改进版：左右图采用相同的绘制风格
    左图 GT 的点值视为 1.0，右图 Pred 的点值为预测概率，统一使用 jet colormap 映射。
    """
    xyz = xyz.cpu().numpy()
    target = target.cpu().numpy()
    pred_prob = pred_prob.cpu().numpy()

    # 1. 提取需要高亮显示的点
    # 预测 mask (大于阈值的点)
    pred_mask = pred_prob > threshold
    pred_points = xyz[pred_mask]
    # 提取这些点的预测概率值用于上色
    pred_colors = pred_prob[pred_mask]

    # 真值 mask (属于路径的点)
    gt_mask = target > 0.5
    gt_points = xyz[gt_mask]
    # 提取这些点的真值用于上色 (它们的值都是 1.0)
    gt_colors = target[gt_mask]

    fig = plt.figure(figsize=(16, 8))

    # --- 定义统一的绘图参数 (关键修改点) ---
    # 左右两边的高亮点都使用这套参数
    scatter_kwargs = {
        'cmap': 'jet',    # 统一使用 jet 颜色映射 (蓝->青->黄->红)
        's': 20,          # 稍微加大点的大小，使其更明显
        'vmin': 0.0,      # 概率最小值
        'vmax': 1.0,      # 概率最大值
        'alpha': 1.0,     # 路径点不透明
        'edgecolor': 'none' #去掉点的边框让颜色更纯粹
    }

    # --- 左图：Ground Truth (真实路径) ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Ground Truth Path (Target=1.0, solid Red)")
    # 画背景点 (灰色，透明度极高，仅作上下文参考)
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='gray', s=1, alpha=0.05)

    # 画真实路径点 - 使用统一风格
    p1 = None
    if len(gt_points) > 0:
        # gt_colors 的值全是 1.0，所以在 jet cmap 下会显示为纯红色
        p1 = ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                         c=gt_colors, label='GT', **scatter_kwargs)

    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    # 设置相同的视角以便对比 (可选)
    ax1.view_init(elev=30, azim=-60)


    # --- 右图：Prediction (模型预测) ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Prediction (Conf > {threshold})")
    # 画背景点 (与左图一致)
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='gray', s=1, alpha=0.05)

    # 画预测路径点 - 使用统一风格
    p2 = None

    if len(pred_points) > 0:
        # pred_colors 是概率值，颜色会根据概率在 jet cmap 上渐变
        p2 = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                         c=pred_colors, label='Pred', **scatter_kwargs)
    
    # 设置相同的视角以便对比 (可选)
    ax2.view_init(elev=30, azim=-60)

    # --- 添加统一的 Colorbar ---
    # 使用 p2 (预测图的句柄) 来生成 colorbar，因为它包含了渐变色
    # 如果 p2 不存在 (没有预测点)，则尝试用 p1
    plot_handle = p2 if p2 is not None else p1
    if plot_handle is not None:
        # 在图的右侧添加一个共享的 colorbar
        cbar = fig.colorbar(plot_handle, ax=[ax1, ax2], shrink=0.7, location='right', pad=0.02)
        cbar.set_label('Probability / Confidence Score')

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
        # 计算真值路径上的平均预测置信度
        path_conf = prob_vis[target_vis==1].mean().item() if (target_vis==1).sum() > 0 else 0

        print(f"\n--- Sample {i} ({os.path.basename(all_files[i])}) ---")
        print(f"Max Predicted Confidence in scene: {max_conf:.4f}")
        print(f"Avg Confidence on GT Path points: {path_conf:.4f}")

        # 可视化 - 阈值可以根据需要调整，例如 0.3 或 0.5
        visualize_result(xyz_vis, target_vis, prob_vis, threshold=0.5)

        # 每看一张图暂停一下，输入 n 退出
        cmd = input("Press Enter for next sample, or 'n' to stop: ")
        if cmd.lower() == 'n':
            break

if __name__ == "__main__":
    # --- 配置路径 (请确保这些路径在你本地是正确的) ---

    # 1. 数据文件夹路径
    DATA_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\raw_model\train_data"

    # 2. 权重文件路径
    # 确保这里指向你训练好的 best_model.pth
    CKPT_PATH = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\raw_model\checkpoints\best_model.pth"

    # 打印路径确认
    print("-" * 50)
    print(f"Data Dir: {DATA_DIR}")
    print(f"Model Path: {CKPT_PATH}")
    print("-" * 50)

    # 执行评估
    evaluate(CKPT_PATH, DATA_DIR)