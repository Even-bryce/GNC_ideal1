import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# 导入你的模块
from src.data.data_loader import PathPointDataset, collate_fn
from src.models.pointnet_transfomer2.my_model import get_model
from torch.utils.data import DataLoader

def extract_waypoints(points_norm, scores, map_dim, eps=0.02, peak_radius=0.02):
    """
    自适应异向空间航路点提取 (归一化进 -> 归一化出)
    """
    if len(points_norm) == 0:
        return np.empty((0, 3))

    Lx, Ly, Lz = map_dim
    
    # 1. 内部换算到 Ratio 空间 (0~1)
    xyz_min = np.array([0.0, 0.0, 0.0])
    xyz_max = np.array([Lx, Ly, Lz])
    center = 0.5 * (xyz_min + xyz_max)
    scale = max(Lx, Ly, Lz)

    points_phys = points_norm * scale + center
    dim_scale = np.array([Lx, Ly, Lz])
    points_ratio = points_phys / dim_scale

    # 2. 局部极大值寻找
    tree = KDTree(points_ratio)
    peaks_ratio = []
    peaks_norm = []  # 同步记录归一化坐标，方便最后直接返回
    peak_scores = []

    for i, p in enumerate(points_ratio):
        idx = tree.query_ball_point(p, r=peak_radius)
        is_peak = True
        for j in idx:
            if scores[j] > scores[i]:
                is_peak = False
                break
        if is_peak:
            peaks_ratio.append(points_ratio[i]) 
            peaks_norm.append(points_norm[i])
            peak_scores.append(scores[i])

    if len(peaks_ratio) == 0:
        return np.empty((0, 3))

    peaks_ratio = np.array(peaks_ratio)
    peaks_norm = np.array(peaks_norm)
    peak_scores = np.array(peak_scores)

    # 3. DBSCAN 聚类
    clustering = DBSCAN(eps=eps, min_samples=1).fit(peaks_ratio)
    labels = clustering.labels_

    waypoints_norm = []
    for label in set(labels):
        if label == -1:
            continue
        mask = (labels == label)
        cluster_norm = peaks_norm[mask]
        cluster_scores = peak_scores[mask]
        
        best_idx = np.argmax(cluster_scores)
        waypoints_norm.append(cluster_norm[best_idx])

    return np.array(waypoints_norm)


# 💡 增加了一个 true_mid_wps 参数接收真实航路点
def visualize_result(xyz, target, pred_prob, start_pt, goal_pt, gt_mid_wps, pred_mid_wps, true_mid_wps, threshold=0.5):
    """
    3D 可视化函数 - 独立绘制起终点，高亮中间航路点，并叠加真实航路点
    """
    xyz = xyz.cpu().numpy()
    target = target.cpu().numpy()
    pred_prob = pred_prob.cpu().numpy()

    pred_mask = pred_prob > threshold
    pred_points = xyz[pred_mask]
    pred_colors = pred_prob[pred_mask]

    gt_mask = target > 0.5
    gt_points = xyz[gt_mask]
    gt_colors = target[gt_mask]

    fig = plt.figure(figsize=(16, 8))
    scatter_kwargs = {
        'cmap': 'jet', 's': 20, 'vmin': 0.0, 'vmax': 1.0, 
        'alpha': 0.5, 'edgecolor': 'none' 
    }

    # --- 左图：Ground Truth ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"Ground Truth Path\n(Found {len(gt_mid_wps)} Mid Waypoints)")
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='gray', s=1, alpha=0.05)

    p1 = None
    if len(gt_points) > 0:
        p1 = ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                         c=gt_colors, label='GT Heatmap', **scatter_kwargs)
        
    ax1.scatter(*start_pt, c='green', s=150, marker='s', edgecolor='black', label='Start')
    ax1.scatter(*goal_pt, c='blue', s=150, marker='s', edgecolor='black', label='Goal')

    # 绘制 从标签热力图提取的 航路点 (红色五角星)
    if len(gt_mid_wps) > 0:
        ax1.scatter(gt_mid_wps[:, 0], gt_mid_wps[:, 1], gt_mid_wps[:, 2],
                    c='red', s=150, marker='*', edgecolor='black', label='Extracted from GT')

    # 💡 绘制 绝对真实的 原始航路点 (黄色大菱形)
    if len(true_mid_wps) > 0:
        ax1.scatter(true_mid_wps[:, 0], true_mid_wps[:, 1], true_mid_wps[:, 2],
                    c='yellow', s=100, marker='D', edgecolor='black', label='True WPs (Raw)')

    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.view_init(elev=30, azim=-60)
    ax1.legend()

    # --- 右图：Prediction ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Prediction (Conf > {threshold})\n(Found {len(pred_mid_wps)} Mid Waypoints)")
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='gray', s=1, alpha=0.05)

    p2 = None
    if len(pred_points) > 0:
        p2 = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                         c=pred_colors, label='Pred Heatmap', **scatter_kwargs)
        
    ax2.scatter(*start_pt, c='green', s=150, marker='s', edgecolor='black', label='Start')
    ax2.scatter(*goal_pt, c='blue', s=150, marker='s', edgecolor='black', label='Goal')

    if len(pred_mid_wps) > 0:
        ax2.scatter(pred_mid_wps[:, 0], pred_mid_wps[:, 1], pred_mid_wps[:, 2],
                    c='magenta', s=150, marker='*', edgecolor='black', label='Pred Mid WPs')
    
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.view_init(elev=30, azim=-60)
    ax2.legend()

    plot_handle = p2 if p2 is not None else p1
    if plot_handle is None:
        plot_handle = ax2.scatter([xyz[0,0]], [xyz[0,1]], [xyz[0,2]], c=[0.0], s=0, **scatter_kwargs)

    cbar = fig.colorbar(plot_handle, ax=[ax1, ax2], shrink=0.7, location='right', pad=0.02)
    cbar.set_label('Probability / Confidence Score')

    plt.show()


def visualize_test(xyz, pred_prob, f_start_b, f_goal_b, start_pt, goal_pt, true_mid_wps=None,
                   delta_s=0.5, R_nms=0.5, K_local=3, delta_d=0.2):
    """
    专门用于验证 get_skeleton_and_pairs 逻辑的三联画可视化函数
    包含：原图与真值对比、骨架点提取结果、有效连线结果，并修复了图例和Colorbar遮挡问题。
    """
    # ==========================================
    # 内部调用：核心的点对筛选逻辑
    # ==========================================
    def get_skeleton_and_pairs(pb, xb, f_start_b, f_goal_b, delta_s=0.5, R_nms=0.5, K_local=3, delta_d=0.2, K_pairs=1):
        """
        K_pairs: 定义每个节点最多寻找的前向/后向邻居数量
        """
        # 💡 终极修复：利用特征场识别起终点
        is_start = f_start_b > 0.95
        is_goal = f_goal_b > 0.95
        
        # 💡 赋予免死金牌：只要预测分数达标，【或者】它是起终点，就允许进入候选池！
        base_mask = (pb > delta_s) | is_start | is_goal 
        
        idx_base = torch.nonzero(base_mask, as_tuple=False).squeeze(-1)
        
        if idx_base.numel() < 2:
            return None, None, None, None, None
            
        pb_base, xb_base = pb[idx_base], xb[idx_base]
        dist_base = torch.cdist(xb_base, xb_base)
        in_nms = dist_base < R_nms
        higher_p = pb_base.unsqueeze(0) < pb_base.unsqueeze(1) 
        higher_count = (in_nms & higher_p).sum(dim=1) 
        v_mask = higher_count < K_local
        V_idx = idx_base[v_mask]
        
        if V_idx.numel() < 2: 
            return None, None, None, None, None
            
        v_pb, v_xb = pb[V_idx], xb[V_idx]
        v_fs, v_fg = f_start_b[V_idx], f_goal_b[V_idx]
        dist_v = torch.cdist(v_xb, v_xb)
        
        # 💡 动态确定 Top-K 的真实数量 (防止筛选后存活的骨架点总数小于 K_pairs 导致报错)
        actual_K = min(K_pairs, V_idx.shape[0])
        
        # 找 k* 集合 (prev)
        cond_k_dist = dist_v > delta_d
        cond_k_fs = v_fs.unsqueeze(0) < v_fs.unsqueeze(1) 
        cond_k_fg = v_fg.unsqueeze(0) > v_fg.unsqueeze(1) 
        valid_k = cond_k_dist & cond_k_fs & cond_k_fg
        diff_fs = v_fs.unsqueeze(1) - v_fs.unsqueeze(0) 
        diff_fs = torch.where(valid_k, diff_fs, torch.full_like(diff_fs, float('inf')))
        
        # 💡 升级为集合：取差值最小的 Top-K 个邻居
        vals_k, k_star_local = torch.topk(diff_fs, k=actual_K, dim=1, largest=False)
        # 只要取出来的最小值不是 inf，就说明找到了合法的真实邻居，has_k 形状变为 [M, actual_K]
        has_k = (vals_k != float('inf'))
        
        # 找 j* 集合 (next)
        cond_j_dist = dist_v > delta_d
        cond_j_fg = v_fg.unsqueeze(0) < v_fg.unsqueeze(1) 
        cond_j_fs = v_fs.unsqueeze(0) > v_fs.unsqueeze(1) 
        valid_j = cond_j_dist & cond_j_fg & cond_j_fs
        diff_fg = v_fg.unsqueeze(1) - v_fg.unsqueeze(0) 
        diff_fg = torch.where(valid_j, diff_fg, torch.full_like(diff_fg, float('inf')))
        
        # 💡 升级为集合：取差值最小的 Top-K 个邻居
        vals_j, j_star_local = torch.topk(diff_fg, k=actual_K, dim=1, largest=False)
        # has_j 形状变为 [M, actual_K]
        has_j = (vals_j != float('inf'))
        
        return V_idx, k_star_local, j_star_local, has_k, has_j
    # 1. 数据准备
    xyz_np = xyz.cpu().numpy()
    pred_prob_np = pred_prob.cpu().numpy()
    
    # 2. 调用筛选逻辑
    V_idx, k_star, j_star, has_k, has_j = get_skeleton_and_pairs(
        pb=pred_prob, xb=xyz, f_start_b=f_start_b, f_goal_b=f_goal_b,
        delta_s=delta_s, R_nms=R_nms, K_local=K_local, delta_d=delta_d
    )

    # 3. 准备画布：1行3列
    fig = plt.figure(figsize=(24, 8))
    scatter_kwargs = {'cmap': 'jet', 's': 15, 'vmin': 0.0, 'vmax': 1.0, 'edgecolor': 'none'}
    bg_kwargs = {'c': 'gray', 's': 1, 'alpha': 0.05}

    # ==========================================
    # Plot 1: 原始预测热力图 + 绝对真值点
    # ==========================================
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title("1. Raw Prediction Heatmap & True WPs")
    
    ax1.scatter(xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2], **bg_kwargs)
    
    vis_mask = pred_prob_np > 0.1
    p1 = None
    if vis_mask.sum() > 0:
        p1 = ax1.scatter(xyz_np[vis_mask, 0], xyz_np[vis_mask, 1], xyz_np[vis_mask, 2], 
                         c=pred_prob_np[vis_mask], **scatter_kwargs)
    else:
        # Fallback 防止 colorbar 报错
        p1 = ax1.scatter([xyz_np[0,0]], [xyz_np[0,1]], [xyz_np[0,2]], c=[0.0], **scatter_kwargs)
        
    ax1.scatter(*start_pt, c='green', s=100, marker='s', edgecolor='black', label='Start')
    ax1.scatter(*goal_pt, c='blue', s=100, marker='s', edgecolor='black', label='Goal')

    if true_mid_wps is not None and len(true_mid_wps) > 0:
        ax1.scatter(true_mid_wps[:, 0], true_mid_wps[:, 1], true_mid_wps[:, 2],
                    c='yellow', s=100, marker='D', edgecolor='black', label='True WPs', zorder=5)

    ax1.view_init(elev=30, azim=-60)

    # ==========================================
    # Plot 2: 筛选出的骨架点 (V_idx)
    # ==========================================
    ax2 = fig.add_subplot(132, projection='3d')
    num_v = len(V_idx) if V_idx is not None else 0
    ax2.set_title(f"2. Skeleton Points (NMS Filtered)\nFound: {num_v} points")
    
    ax2.scatter(xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2], **bg_kwargs)
    ax2.scatter(*start_pt, c='green', s=100, marker='s', edgecolor='black', label='Start')
    ax2.scatter(*goal_pt, c='blue', s=100, marker='s', edgecolor='black', label='Goal')

    if V_idx is not None and num_v > 0:
        v_idx_np = V_idx.cpu().numpy()
        v_xyz_np = xyz_np[v_idx_np]
        v_prob_np = pred_prob_np[v_idx_np]
        
        ax2.scatter(v_xyz_np[:, 0], v_xyz_np[:, 1], v_xyz_np[:, 2], 
                    c=v_prob_np, s=50, marker='o', edgecolor='black', cmap='jet', vmin=0, vmax=1)

    ax2.view_init(elev=30, azim=-60)

    # ==========================================
    # Plot 3: 连线点对 (Valid Pairs & Edges)
    # ==========================================
    ax3 = fig.add_subplot(133, projection='3d')
    
    ax3.scatter(xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2], **bg_kwargs)
    ax3.scatter(*start_pt, c='green', s=100, marker='s', edgecolor='black', label='Start')
    ax3.scatter(*goal_pt, c='blue', s=100, marker='s', edgecolor='black', label='Goal')

    valid_edges = 0
    if V_idx is not None and num_v > 0:
        k_star_np = k_star.cpu().numpy()
        j_star_np = j_star.cpu().numpy()
        has_k_np = has_k.cpu().numpy()
        has_j_np = has_j.cpu().numpy()

        ax3.scatter(v_xyz_np[:, 0], v_xyz_np[:, 1], v_xyz_np[:, 2], 
                    c='black', s=20, marker='o', zorder=4)

        for i in range(num_v):
            pt_i = v_xyz_np[i]
            
            # 遍历 K 个可能的后向邻居 (prev)
            for m in range(k_star_np.shape[1]):
                if has_k_np[i, m]:  # 注意这里变成了二维索引 [i, m]
                    pt_k = v_xyz_np[k_star_np[i, m]]
                    ax3.plot([pt_i[0], pt_k[0]], [pt_i[1], pt_k[1]], [pt_i[2], pt_k[2]], 
                             color='darkorange', linewidth=2, alpha=0.8, zorder=3)
                    valid_edges += 1
                    
            # 遍历 K 个可能的前向邻居 (next)
            for m in range(j_star_np.shape[1]):
                if has_j_np[i, m]:  # 注意这里变成了二维索引 [i, m]
                    pt_j = v_xyz_np[j_star_np[i, m]]
                    ax3.plot([pt_i[0], pt_j[0]], [pt_i[1], pt_j[1]], [pt_i[2], pt_j[2]], 
                             color='purple', linewidth=2, alpha=0.8, zorder=3)
                    valid_edges += 1

    ax3.set_title(f"3. Valid Pairs & Edges (Loss applies here)\nTotal Edges: {valid_edges}")
    ax3.view_init(elev=30, azim=-60)

    ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), framealpha=0.9)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), framealpha=0.9)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), framealpha=0.9)

    # 2. 压缩 3D 子图占用空间，为【左侧】 Colorbar 留出白边
    # 把 left 从 0.02 增加到 0.08，把整体画面往右挤一挤
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.95, wspace=0.05)

    # 3. 在最左侧安全区域手动划定 Colorbar 专属地盘 [左, 下, 宽, 高]
    # 把起始横坐标从 0.94(最右) 改成 0.02(最左)
    cbar_ax = fig.add_axes([0.02, 0.25, 0.015, 0.5]) 
    cbar = fig.colorbar(p1, cax=cbar_ax)
    
    # 💡 细节优化：把颜色条的刻度和标签移到左边，看着更顺眼
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_label('Confidence Score (prob)')

    plt.show()

def filter_zigzag_waypoints(start_pt, goal_pt, mid_wps, min_dist=0.03, local_thresh=0.15, max_turn_angle=60.0):
    """
    对聚类提取出的无序航路点进行排序和去曲折筛选 (局部 Z-jitter 消除版)。
    
    参数:
        start_pt, goal_pt: 起点和终点坐标 (1D numpy array)
        mid_wps: 聚类提取出的中间航路点集合 (N x 3 numpy array)
        min_dist: 最小容忍距离 (如 0.03)，绝对重叠的点直接剔除
        local_thresh: 局部区域阈值 (如 0.15)，只有距离小于此值的点，才去校验其转角
        max_turn_angle: 最大容忍转角 (度)，局部区域内大于此角度认为是抖动，直接剔除
    """
    if len(mid_wps) == 0:
        return mid_wps

    # ==========================================
    # Step 1: 利用 f_goal 相对比例进行全局排序
    # ==========================================
    d_s = np.linalg.norm(mid_wps - start_pt, axis=1)
    d_g = np.linalg.norm(mid_wps - goal_pt, axis=1)
    f_goal = d_s / (d_s + d_g + 1e-6)
    
    sorted_indices = np.argsort(f_goal)
    ordered_wps = mid_wps[sorted_indices]

    # ==========================================
    # Step 2: 局部曲折剪枝 (Local Zig-Zag Pruning)
    # ==========================================
    path = [start_pt] + list(ordered_wps) + [goal_pt]
    
    i = 1
    while i < len(path) - 1:
        prev_pt = path[i-1]
        curr_pt = path[i]
        next_pt = path[i+1]

        v_in = curr_pt - prev_pt
        v_out = next_pt - curr_pt
        n_in = np.linalg.norm(v_in)
        n_out = np.linalg.norm(v_out)

        # 规则 1：绝对距离过近 (几乎重叠的冗余点)
        if n_in < min_dist or n_out < min_dist:
            path.pop(i)
            continue

        # 规则 2：先看距离，再看角度！(你的核心改进)
        if n_in > 1e-6 and n_out > 1e-6:
            # 判断当前点是否处于一个“局部小碎步”状态
            # 如果进入该点或离开该点的步长很短，说明这是一个局部细节
            if n_in < local_thresh or n_out < local_thresh:
                
                # 只有在局部范围内，才去计算转角
                cos_theta = np.dot(v_in, v_out) / (n_in * n_out)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                turn_angle = np.degrees(np.arccos(cos_theta))
                
                # 如果局部范围内发生了剧烈折返，认定为 Z 字形抖动，剔除
                if turn_angle > max_turn_angle:
                    path.pop(i)
                    continue

        i += 1

    return np.array(path[1:-1])

def calculate_path_length_by_feature(start_pt, goal_pt, mid_wps, scale, center):
    """
    利用类似网络特征索引 4 (f_goal) 的几何逻辑，
    对散乱的中间航路点进行排序，并计算从起点到终点的物理折线距离。
    """
    if mid_wps is None or len(mid_wps) == 0:
        # 如果没有中间点，直接算起点到终点的直线距离
        path_norm = np.vstack([start_pt, goal_pt])
    else:
        # 重新计算类似于“特征索引 4”的 f_goal 值用于排序
        d_s = np.linalg.norm(mid_wps - start_pt, axis=1)
        d_g = np.linalg.norm(mid_wps - goal_pt, axis=1)
        f_goal = d_s / (d_s + d_g + 1e-8)  # 越靠近终点，值越接近 1
        
        # 按 f_goal 升序排列 (即从起点顺藤摸瓜到终点)
        sorted_indices = np.argsort(f_goal)
        ordered_mid_wps = mid_wps[sorted_indices]
        
        # 拼接完整路径: 起点 -> 排序后的中间点 -> 终点
        path_norm = np.vstack([start_pt, ordered_mid_wps, goal_pt])
        
    # 将归一化坐标还原为真实物理坐标 (利用 scale 和 center)
    path_phys = path_norm * scale + center
    
    # 计算相邻点之间的欧氏距离并求和
    diffs = path_phys[1:] - path_phys[:-1]
    return np.sum(np.linalg.norm(diffs, axis=1))

@torch.no_grad()


@torch.no_grad()
def evaluate(model_path, data_dir, map_dim, cluster_eps=0.02, peak_radius=0.02):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(all_files) == 0:
        print(f"Error: 在该目录下找不到 .npz 数据 -> {data_dir}")
        return

    dataset = PathPointDataset(all_files)
    sample_points, sample_labels, sample_waypoints = dataset[0]
    real_input_dim = sample_points.shape[1]

    model = get_model(num_classes=1, input_dim=real_input_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 如果你想测试整个目录，split可以设为0；如果是验证集划分，按你原来的逻辑保留即可
    split = int(len(all_files) * 0.8)
    val_files = all_files[split:]

    dataset = PathPointDataset(val_files)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ==========================================
    # 💡 全局评估累加器
    # ==========================================
    total_true_dist = 0.0
    total_gt_dist = 0.0
    total_pred_dist = 0.0
    evaluated_samples = 0

    print(f"🚀 开始批量评估，共计 {len(val_files)} 个样本，请稍候...")

    for i, (points, targets, gt_waypoints_list) in enumerate(test_loader):
        points = points.to(device)   
        targets = targets.to(device) 
        points_trans = points.permute(0, 2, 1).contiguous() 

        output = model(points_trans)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        probs = torch.sigmoid(logits) 

        # 1. 取出当前样本的数据
        xyz_vis = points[0, :, :3]      
        target_vis = targets[0, :, 0]   
        prob_vis = probs[0, :, 0]       

        start_pt = xyz_vis[0].cpu().numpy()
        goal_pt = xyz_vis[1].cpu().numpy()

        mid_xyz = xyz_vis[2:]
        mid_target = target_vis[2:]
        mid_prob = prob_vis[2:]

        # 2. 获取原始的 True Waypoints (完全真实的航路点)
        raw_wps_phys = gt_waypoints_list[0]
        if isinstance(raw_wps_phys, torch.Tensor):
            raw_wps_phys = raw_wps_phys.cpu().numpy()
            
        Lx, Ly, Lz = map_dim
        center = 0.5 * np.array([Lx, Ly, Lz])
        scale = max(Lx, Ly, Lz)
        raw_wps_norm = (raw_wps_phys - center) / scale
        
        if len(raw_wps_norm) > 2:
            true_mid_wps = raw_wps_norm[1:-1]
        else:
            true_mid_wps = np.empty((0, 3))

        # 3. 预测热力图聚类与过滤
        pred_mask = mid_prob > 0.7
        pred_xyz_filtered = mid_xyz[pred_mask].cpu().numpy()
        pred_scores_filtered = mid_prob[pred_mask].cpu().numpy()
        
        raw_pred_mid_waypoints = extract_waypoints(
            pred_xyz_filtered, pred_scores_filtered, 
            map_dim=map_dim, eps=cluster_eps, peak_radius=peak_radius
        )

        pred_mid_waypoints = filter_zigzag_waypoints(
            start_pt, goal_pt, raw_pred_mid_waypoints, 
            min_dist=0.05,       
            local_thresh=0.2,    
            max_turn_angle=60.0  
        )
        
        # 4. GT热力图聚类 (仅供对比参考)
        gt_mask = mid_target > 0.7
        gt_xyz_filtered = mid_xyz[gt_mask].cpu().numpy()
        gt_scores_filtered = mid_target[gt_mask].cpu().numpy()
        
        gt_mid_waypoints = extract_waypoints(
            gt_xyz_filtered, gt_scores_filtered, 
            map_dim=map_dim, eps=cluster_eps, peak_radius=peak_radius
        )


        if i <= 10:
            # 调用画图
            # visualize_result(
            #     xyz_vis, target_vis, prob_vis, 
            #     start_pt, goal_pt, gt_mid_waypoints, pred_mid_waypoints, true_mid_wps,
            #     threshold=0.5
            # )
            f_start_b = points[0, :, 3]  # 取出所有点的 f_start
            f_goal_b  = points[0, :, 4]  # 取出所有点的 f_goal

            visualize_test(
                xyz=xyz_vis, 
                pred_prob=prob_vis, 
                f_start_b=f_start_b, 
                f_goal_b=f_goal_b, 
                start_pt=start_pt, 
                goal_pt=goal_pt,
                true_mid_wps=raw_wps_norm,
                delta_s=0.8,       # 替换为你的真实参数
                R_nms=0.15,        # 替换为你的真实参数
                K_local=3,         # 替换为你的真实参数
                delta_d=0.2        # 替换为你的真实参数
            )

            cmd = input("Press Enter for next sample, or 'n' to stop: ")
            if cmd.lower() == 'n':
                break

        # 5. 计算三种路径的物理距离
        true_path_dist = calculate_path_length_by_feature(start_pt, goal_pt, true_mid_wps, scale, center)
        gt_cluster_dist = calculate_path_length_by_feature(start_pt, goal_pt, gt_mid_waypoints, scale, center)
        pred_cluster_dist = calculate_path_length_by_feature(start_pt, goal_pt, pred_mid_waypoints, scale, center)

        # 累加
        total_true_dist += true_path_dist
        total_gt_dist += gt_cluster_dist
        total_pred_dist += pred_cluster_dist
        evaluated_samples += 1

        # 可选：打印简单的进度条，防止看着像死机了
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1} / {len(val_files)}...")

    # ==========================================
    # 💡 最终测试集平均结果输出
    # ==========================================
    if evaluated_samples > 0:
        avg_true_dist = total_true_dist / evaluated_samples
        avg_gt_dist = total_gt_dist / evaluated_samples
        avg_pred_dist = total_pred_dist / evaluated_samples
        
        print("\n" + "="*50)
        print(f"🏁 测试完成！总共评估了 {evaluated_samples} 个样本")
        print("="*50)
        print(f"👑 绝对真值路径 (Raw WPs) 平均长度 : {avg_true_dist:.2f}")
        print(f"🎯 标签热力图聚类 (GT)   平均长度 : {avg_gt_dist:.2f}")
        print(f"🤖 模型预测热力图 (Pred) 平均长度 : {avg_pred_dist:.2f}")
        
        # 计算一下预测路径相对于真实路径的长度偏差比例
        if avg_true_dist > 0:
            diff_ratio = ((avg_pred_dist - avg_true_dist) / avg_true_dist) * 100
            print("-" * 50)
            print(f"📈 预测路径相比绝对真值，平均长度偏差: {diff_ratio:+.2f}%")
        print("="*50 + "\n")

# def evaluate(model_path, data_dir, map_dim, cluster_eps=0.02, peak_radius=0.02):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
#     if len(all_files) == 0:
#         print(f"Error: 在该目录下找不到 .npz 数据 -> {data_dir}")
#         return

#     dataset = PathPointDataset(all_files)
#     sample_points, sample_labels, sample_waypoints = dataset[0]
#     real_input_dim = sample_points.shape[1]

#     model = get_model(num_classes=1, input_dim=real_input_dim).to(device)
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint)
#     model.eval()

#     split = int(len(all_files) * 0)
#     val_files = all_files[split:]

#     dataset = PathPointDataset(val_files)
#     test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     for i, (points, targets, gt_waypoints_list) in enumerate(test_loader):
#         points = points.to(device)   
#         targets = targets.to(device) 

#         points_trans = points.permute(0, 2, 1).contiguous() 

#         output = model(points_trans)
#         logits = output[0] if isinstance(output, (tuple, list)) else output
#         probs = torch.sigmoid(logits) 

#         # 1. 取出当前样本的数据 (都是归一化数据)
#         xyz_vis = points[0, :, :3]      
#         target_vis = targets[0, :, 0]   
#         prob_vis = probs[0, :, 0]       

#         start_pt = xyz_vis[0].cpu().numpy()
#         goal_pt = xyz_vis[1].cpu().numpy()

#         # 剔除起终点用于后续聚类预测
#         mid_xyz = xyz_vis[2:]
#         mid_target = target_vis[2:]
#         mid_prob = prob_vis[2:]

#         # ==========================================
#         # 💡 新增：处理绝对真实的原始航路点
#         # ==========================================
#         raw_wps_phys = gt_waypoints_list[0]
#         if isinstance(raw_wps_phys, torch.Tensor):
#             raw_wps_phys = raw_wps_phys.cpu().numpy()
            
#         # 归一化真实航路点，以匹配可视化坐标系
#         Lx, Ly, Lz = map_dim
#         center = 0.5 * np.array([Lx, Ly, Lz])
#         scale = max(Lx, Ly, Lz)
#         raw_wps_norm = (raw_wps_phys - center) / scale
        
#         # 剔除起终点 (保存时的第0个是起点，最后一个是终点)
#         if len(raw_wps_norm) > 2:
#             true_mid_wps = raw_wps_norm[1:-1]
#         else:
#             true_mid_wps = np.empty((0, 3))


#         # --- 预测航路点聚类 ---
#         pred_mask = mid_prob > 0.7
#         pred_xyz_filtered = mid_xyz[pred_mask].cpu().numpy()
#         pred_scores_filtered = mid_prob[pred_mask].cpu().numpy()
        
#         raw_pred_mid_waypoints = extract_waypoints(
#             pred_xyz_filtered, pred_scores_filtered, 
#             map_dim=map_dim, eps=cluster_eps, peak_radius=peak_radius
#         )

#         pred_mid_waypoints = filter_zigzag_waypoints(
#             start_pt, goal_pt, raw_pred_mid_waypoints, 
#             min_dist=0.05,        # 距离阈值，可根据你的尺度微调
#             local_thresh=0.2,   # 局部范围阈值，越大越宽松
#             max_turn_angle=60.0   # 角度阈值，越小越趋近于拉直
#         )
        
#         # --- 真值热力图航路点聚类 ---
#         gt_mask = mid_target > 0.9
#         gt_xyz_filtered = mid_xyz[gt_mask].cpu().numpy()
#         gt_scores_filtered = mid_target[gt_mask].cpu().numpy()
        
#         gt_mid_waypoints = extract_waypoints(
#             gt_xyz_filtered, gt_scores_filtered, 
#             map_dim=map_dim, eps=cluster_eps, peak_radius=peak_radius
#         )

#         print(f"\n--- Sample {i} ({os.path.basename(val_files[i])}) ---")
#         print(f"Max Conf: {prob_vis.max().item():.4f}")
#         print(f"Extracted GT Mid Waypoints: {len(gt_mid_waypoints)}")
#         print(f"Extracted Pred Mid Waypoints: {len(pred_mid_waypoints)}")
#         print(f"Raw True Mid Waypoints: {len(true_mid_wps)}")

#         # 调用画图
#         visualize_result(
#             xyz_vis, target_vis, prob_vis, 
#             start_pt, goal_pt, gt_mid_waypoints, pred_mid_waypoints, true_mid_wps,
#             threshold=0.5
#         )

#         cmd = input("Press Enter for next sample, or 'n' to stop: ")
#         if cmd.lower() == 'n':
#             break

if __name__ == "__main__":
    
    DATA_DIR = r"C:\Users\Administrator\Desktop\experiments\train_data6"
    # DATA_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\src\data\data_for_train\train_data4"
    CKPT_PATH = r"C:\Users\Administrator\Desktop\experiments\checkpoints\ckpt_epoch_40.pth"



    # ==========================================
    # 超参数控制台
    # ==========================================
    MAP_DIM = np.array([1500.0, 1500.0, 240.0]) 

    # 直接使用比例空间参数 (例如 0.15 代表整个地图 15% 的跨度)
    CLUSTER_EPS = 0.15 
    PEAK_RADIUS = 0.15 

    evaluate(
        model_path=CKPT_PATH, 
        data_dir=DATA_DIR, 
        map_dim=MAP_DIM,
        cluster_eps=CLUSTER_EPS,
        peak_radius=PEAK_RADIUS
    )