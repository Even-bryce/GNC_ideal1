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
    # dataset[0] 返回的是单条数据，没有经过 collate_fn，所以还是 3 个变量，这里不用改
    sample_points, sample_labels, sample_waypoints = dataset[0]
    real_input_dim = sample_points.shape[1]

    model = get_model(num_classes=1, input_dim=real_input_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    split = int(len(all_files) * 0.8)
    val_files = all_files[split:]

    dataset = PathPointDataset(val_files)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ==========================================
    # 💡 物理距离评估累加器
    # ==========================================
    total_true_dist = 0.0
    total_gt_dist = 0.0
    total_pred_dist = 0.0
    evaluated_samples = 0

    # ==========================================
    # 💡 分类概率评估累加器
    # ==========================================
    val_max_probs, val_neg_max_probs, val_neg_fpr_list = [], [], []
    val_se_mean_probs, val_mid_mean_probs, val_mid_fnr_list = [], [], []

    print(f"🚀 开始批量评估，共计 {len(val_files)} 个样本，请稍候...")

    with torch.no_grad():
        # 💡 修改 1：解包时接收 mask
        for i, (points, targets, mask, gt_waypoints_list) in enumerate(test_loader):
            points = points.to(device)   
            targets = targets.to(device) 
            mask = mask.to(device) # 💡 将 mask 放入 GPU
            
            points_trans = points.permute(0, 2, 1).contiguous() 

            # 💡 修改 2：把 mask 传给模型
            output = model(points_trans, mask=mask)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            probs = torch.sigmoid(logits) 

            # ==========================================
            # 💡 修改 3：统计分类指标时，全部加上 `& mask` 过滤假点
            # ==========================================
            probs_sq = probs.squeeze(-1) if probs.dim() == 3 else probs       # [B, N]
            targets_sq = targets.squeeze(-1) if targets.dim() == 3 else targets # [B, N]

            if mask.sum() > 0:
                val_max_probs.append(probs_sq[mask].max().item())
            
            # --- 统计误检 (FPR) ---
            neg_mask = (targets_sq < 0.1) & mask # 💡 加上 mask
            if neg_mask.sum() > 0:
                neg_probs = probs_sq[neg_mask]
                val_neg_max_probs.append(neg_probs.max().item())
                num_false_pos = (neg_probs > 0.5).float().sum()
                fpr = num_false_pos / neg_probs.numel()
                val_neg_fpr_list.append(fpr.item())

            # --- 统计漏检 (FNR) 及 SE/MID 概率 ---
            pos_mask = (targets_sq > 0.8) & mask # 💡 加上 mask
            if points_trans.shape[1] >= 6:  
                f_start = points_trans[:, 3, :]  
                f_goal  = points_trans[:, 4, :]
                is_start_end = (f_start > 0.95) | (f_goal > 0.95)
                mask_SE = pos_mask & is_start_end       
                mask_MID = pos_mask & (~is_start_end)   

                if mask_SE.sum() > 0:
                    val_se_mean_probs.append(probs_sq[mask_SE].mean().item())
                if mask_MID.sum() > 0:
                    mid_probs = probs_sq[mask_MID]
                    val_mid_mean_probs.append(mid_probs.mean().item())
                    num_false_neg = (mid_probs < 0.5).float().sum()
                    fnr = num_false_neg / mid_probs.numel()
                    val_mid_fnr_list.append(fnr.item())

            # ==========================================
            # 💡 修改 4：可视化与聚类前，彻底剔除 Padding 数据
            # ==========================================
            # 获取当前 batch 的第 0 个样本 (因为 test_loader batch_size=1)
            xyz_vis_raw = points[0, :, :3]      
            target_vis_raw = targets[0, :, 0] if targets.dim() == 3 else targets[0, :]
            prob_vis_raw = probs[0, :, 0] if probs.dim() == 3 else probs[0, :]
            mask_vis = mask[0, :] 

            # 💡 核心：只保留有效点 (True) 的数据，彻底切断 (0,0,0) 的干扰
            xyz_vis = xyz_vis_raw[mask_vis]
            target_vis = target_vis_raw[mask_vis]
            prob_vis = prob_vis_raw[mask_vis]

            # 经过过滤后，前两个有效点肯定是起点和终点
            start_pt = xyz_vis[0].cpu().numpy()
            goal_pt = xyz_vis[1].cpu().numpy()

            mid_xyz = xyz_vis[2:]
            mid_target = target_vis[2:]
            mid_prob = prob_vis[2:]

            # ==========================================
            # 后面原有的逻辑保持不变
            # ==========================================
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

            pred_mask_vis = mid_prob > 0.8
            pred_xyz_filtered = mid_xyz[pred_mask_vis].cpu().numpy()
            pred_scores_filtered = mid_prob[pred_mask_vis].cpu().numpy()
            
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
            
            gt_mask_vis = mid_target > 0.8
            gt_xyz_filtered = mid_xyz[gt_mask_vis].cpu().numpy()
            gt_scores_filtered = mid_target[gt_mask_vis].cpu().numpy()
            
            gt_mid_waypoints = extract_waypoints(
                gt_xyz_filtered, gt_scores_filtered, 
                map_dim=map_dim, eps=cluster_eps, peak_radius=peak_radius
            )

            if 0 <= i <= 10:
                visualize_result(
                    xyz_vis, target_vis, prob_vis, 
                    start_pt, goal_pt, gt_mid_waypoints, pred_mid_waypoints, true_mid_wps,
                    threshold=0.5
                )

                cmd = input("Press Enter for next sample, or 'n' to stop: ")
                if cmd.lower() == 'n':
                    break

            true_path_dist = calculate_path_length_by_feature(start_pt, goal_pt, true_mid_wps, scale, center)
            gt_cluster_dist = calculate_path_length_by_feature(start_pt, goal_pt, gt_mid_waypoints, scale, center)
            pred_cluster_dist = calculate_path_length_by_feature(start_pt, goal_pt, pred_mid_waypoints, scale, center)

            total_true_dist += true_path_dist
            total_gt_dist += gt_cluster_dist
            total_pred_dist += pred_cluster_dist
            evaluated_samples += 1

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1} / {len(val_files)}...")

    # ==========================================
    # 💡 最终测试集平均结果输出 (保持原样)
    # ==========================================
    if evaluated_samples > 0:
        # 计算距离均值
        avg_true_dist = total_true_dist / evaluated_samples
        avg_gt_dist = total_gt_dist / evaluated_samples
        avg_pred_dist = total_pred_dist / evaluated_samples
        
        avg_neg_max = np.mean(val_neg_max_probs) if len(val_neg_max_probs) > 0 else 0.0
        avg_fpr = np.mean(val_neg_fpr_list) if len(val_neg_fpr_list) > 0 else 0.0
        avg_se_mean = np.mean(val_se_mean_probs) if len(val_se_mean_probs) > 0 else 0.0
        avg_mid_mean = np.mean(val_mid_mean_probs) if len(val_mid_mean_probs) > 0 else 0.0
        avg_mid_fnr = np.mean(val_mid_fnr_list) if len(val_mid_fnr_list) > 0 else 0.0
        
        print("\n" + "="*60)
        print(f"🏁 验证集测试完成！共计评估 {evaluated_samples} 个样本")
        print("="*60)
        
        print("📊 [预测概率与分类性能]")
        print(f"   ➤ SE (起终点) 平均置信度 : {avg_se_mean:.3f}")
        print(f"   ➤ Mid (中间点) 平均置信度: {avg_mid_mean:.3f}")
        print(f"   ➤ NegMax (负样本最高概率): {avg_neg_max:.3f}")
        print(f"   ➤ FPR (误检率 / 假阳性)  : {avg_fpr:.4f}  <-- 越低说明产生的无用废点越少")
        print(f"   ➤ FNR (漏检率 / 假阴性)  : {avg_mid_fnr:.4f}  <-- 越低说明真实路径找得越全")
        print("-" * 60)
        
        print("📏 [物理路径长度表现]")
        print(f"   👑 绝对真值 (Raw WPs) 平均长度 : {avg_true_dist:.2f}")
        print(f"   🎯 标签聚类 (GT)      平均长度 : {avg_gt_dist:.2f}")
        print(f"   🤖 模型预测 (Pred)    平均长度 : {avg_pred_dist:.2f}")
        
        if avg_true_dist > 0:
            diff_ratio = ((avg_pred_dist - avg_true_dist) / avg_true_dist) * 100
            print("-" * 60)
            print(f"   📈 预测相比绝对真值，平均长度偏差: {diff_ratio:+.2f}%")
        print("="*60 + "\n")

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
    
    # DATA_DIR = r"C:\Users\Administrator\Desktop\experiments\train_data9"
    DATA_DIR = r"C:\Users\Administrator\Desktop\experiments\train_data5"
    # DATA_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\src\data\data_for_train\train_data4"
    # CKPT_PATH = r"C:\Users\Administrator\Desktop\experiments\checkpoints\best_model.pth"
    CKPT_PATH = r"C:\Users\Administrator\Desktop\experiments\best_model_for_trian_data5\best_model.pth"



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