import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

def extract_waypoints(points, scores, eps=0.05, peak_radius=0.05, z_weight=1.0):
    if len(points) == 0:
        return np.empty((0, 3))

    # ================================
    # ⭐ Step 0: Z方向加权（拉伸Z轴距离，切断上下层连通）
    # ================================
    points_scaled = points.copy()
    points_scaled[:, 2] *= z_weight

    # ================================
    # Step 1: 局部极大值 (寻找峰值候选点)
    # ================================
    tree = KDTree(points_scaled)
    peaks = []
    peak_scores = []

    for i, p in enumerate(points_scaled):
        # 找周围半径内的邻居
        idx = tree.query_ball_point(p, r=peak_radius)

        is_peak = True
        for j in idx:
            # 如果周围有比自己得分严格更高的点，那自己就不是局部的绝对波峰
            if scores[j] > scores[i]:
                is_peak = False
                break

        if is_peak:
            peaks.append(points[i])  # ⚠️ 记录原始坐标
            peak_scores.append(scores[i])

    if len(peaks) == 0:
        return np.empty((0, 3))

    peaks = np.array(peaks)
    peak_scores = np.array(peak_scores)

    # ================================
    # ⭐ Step 2: DBSCAN 聚类（融合相近的局部极值点）
    # ================================
    peaks_scaled = peaks.copy()
    peaks_scaled[:, 2] *= z_weight

    clustering = DBSCAN(eps=eps, min_samples=1).fit(peaks_scaled)
    labels = clustering.labels_

    waypoints = []

    for label in set(labels):
        if label == -1:
            continue

        mask = (labels == label)
        cluster_points = peaks[mask]
        cluster_scores = peak_scores[mask]

        # 在同属于一个波峰簇的候选点中，选得分最高的那一个作为最终航路点
        best_idx = np.argmax(cluster_scores)
        waypoints.append(cluster_points[best_idx])

    return np.array(waypoints)

def visualize_gt_only(xyz, target, start_pt, goal_pt, gt_mid_wps, threshold=0.7):
    fig = plt.figure(figsize=(16, 8))
    
    scatter_kwargs = {
        'cmap': 'jet', 's': 20, 'vmin': 0.0, 'vmax': 1.0, 
        'alpha': 0.6, 'edgecolor': 'none' 
    }

    # --- 左图：全量真值点云分布 ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Raw GT Data Distribution\n(All Points)")
    p1 = ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=target, **scatter_kwargs)
    
    ax1.scatter(*start_pt, c='green', s=150, marker='s', edgecolor='black', label='Start')
    ax1.scatter(*goal_pt, c='blue', s=150, marker='s', edgecolor='black', label='Goal')

    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.view_init(elev=30, azim=-60)
    ax1.legend()

    # --- 右图：高分点云 + 聚类航路点 ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Filtered GT (Score > {threshold})\nFound {len(gt_mid_wps)} Mid Waypoints")
    
    # 画底色灰点作参考
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='gray', s=1, alpha=0.05)

    # 提取并绘制大于阈值的点
    mask = target > threshold
    high_score_xyz = xyz[mask]
    high_score_target = target[mask]
    
    if len(high_score_xyz) > 0:
        ax2.scatter(high_score_xyz[:, 0], high_score_xyz[:, 1], high_score_xyz[:, 2],
                    c=high_score_target, **scatter_kwargs)
        
    ax2.scatter(*start_pt, c='green', s=150, marker='s', edgecolor='black', label='Start')
    ax2.scatter(*goal_pt, c='blue', s=150, marker='s', edgecolor='black', label='Goal')

    # 绘制聚类提取出的航路点
    if len(gt_mid_wps) > 0:
        ax2.scatter(gt_mid_wps[:, 0], gt_mid_wps[:, 1], gt_mid_wps[:, 2],
                    c='red', s=200, marker='*', edgecolor='black', label='Extracted WPs')
    
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.view_init(elev=30, azim=-60)
    ax2.legend()

    # --- Colorbar ---
    cbar = fig.colorbar(p1, ax=[ax1, ax2], shrink=0.7, location='right', pad=0.02)
    cbar.set_label('Ground Truth Score')
    plt.show()

def evaluate_gt_only(data_dir, cluster_eps=0.05, peak_radius=0.05, score_threshold=0.8):
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(all_files) == 0:
        print(f"Error: 找不到 .npz 数据 -> {data_dir}")
        return

    split = int(len(all_files) * 0)
    val_files = all_files[split:]

    print(f"Total files: {len(all_files)}, Testing on {len(val_files)} files.")

    for i, file_path in enumerate(val_files):
        data = np.load(file_path)
        points = data['points']
        labels = data['labels']
        
        xyz = points[:, :3]
        target = labels[:, 0]

        start_pt = xyz[0]
        goal_pt = xyz[1]

        mid_xyz = xyz[2:]
        mid_target = target[2:]

        gt_mask = mid_target > score_threshold
        gt_xyz_filtered = mid_xyz[gt_mask]
        gt_scores_filtered = mid_target[gt_mask]
        
        gt_mid_waypoints = extract_waypoints(
            gt_xyz_filtered, 
            gt_scores_filtered, 
            eps=cluster_eps, 
            peak_radius=peak_radius,
            z_weight=15.0  # 沿用你评估函数里写的 15.0
        )

        print(f"\n--- Sample {i} ({os.path.basename(file_path)}) ---")
        print(f"Max GT Score in scene: {target.max():.4f}")
        print(f"Points > {score_threshold}: {len(gt_xyz_filtered)}")
        print(f"Extracted GT Mid Waypoints: {len(gt_mid_waypoints)}")

        visualize_gt_only(
            xyz, target, 
            start_pt, goal_pt, gt_mid_waypoints, 
            threshold=score_threshold
        )

        cmd = input("Press Enter for next sample, or 'n' to stop: ")
        if cmd.lower() == 'n':
            break

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\src\data\data_for_train\train_data3"

    # =======================================================
    # 完美匹配归一化坐标的参数配置！
    # =======================================================
    SCORE_THRESHOLD = 0.8  # 只给得分大于 0.8 的点做局部极值判定
    PEAK_RADIUS = 0.25      # 找极值点的视野半径
    CLUSTER_EPS = 0.25      # DBSCAN 合并相邻极值点的半径 (⚠️ 之前这里是 20.0，已被修正)

    evaluate_gt_only(
        DATA_DIR, 
        cluster_eps=CLUSTER_EPS, 
        peak_radius=PEAK_RADIUS, 
        score_threshold=SCORE_THRESHOLD
    )