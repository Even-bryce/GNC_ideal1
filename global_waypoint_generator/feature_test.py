# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import glob
from scipy.spatial import KDTree
from tqdm import tqdm

def visualize_feature_distributions(npz_file_path, save_dir='./output_plots'):
    print(f"正在加载数据: {npz_file_path}")
    data = np.load(npz_file_path)
    points = data['points']  # shape: [N, 9]
    labels = data['labels']  # shape: [N, 2]

    feature_names = ['f_start', 'f_goal', 'd_obs_norm', 'nx', 'ny']
    
    # 1. 提取有效数据
    features = points[2:, 3:8]
    point_scores = labels[2:, 0] 

    # 2. 划分正负样本
    wp_mask = point_scores > 0.8
    bg_mask = point_scores < 0.4

    wp_features = features[wp_mask]
    bg_features = features[bg_mask]

    print(f"提取到正样本(航路点附近): {len(wp_features)} 个")
    print(f"提取到负样本(背景点): {len(bg_features)} 个")

    if len(wp_features) == 0:
        print("该样本中没有得分 > 0.8 的非起终点数据，请检查标签生成逻辑或降低阈值。")
        return

    # 3. 构建 DataFrame
    df_wp = pd.DataFrame(wp_features, columns=feature_names)
    df_wp['Class'] = 'Waypoint (Score > 0.8)'

    df_bg = pd.DataFrame(bg_features, columns=feature_names)
    df_bg['Class'] = 'Background (Score < 0.4)'

    df_all = pd.concat([df_wp, df_bg], ignore_index=True)

    # 4. 绘制特征的一维概率密度分布
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        sns.histplot(
            data=df_all, 
            x=feat,               
            hue='Class',          
            stat="density",       
            common_norm=False,    
            kde=True,             
            alpha=0.4,            
            ax=axes[i],
            palette=['#FF5722', '#03A9F4'] 
        )
        
        axes[i].set_title(f'Distribution of {feat}')
        axes[i].set_xlabel("Feature Value")
        axes[i].set_ylabel("Density")

    plt.suptitle("Feature Distribution Analysis", fontsize=16)
    plt.tight_layout()

    # ==========================================
    # 💡 修改点 2：保存图片逻辑 (必须在 plt.show 之前)
    # ==========================================
    # 确保保存的文件夹存在，如果不存在则自动创建
    os.makedirs(save_dir, exist_ok=True)
    
    # 构造唯一的文件名（这里提取原始 npz 的名字，加上后缀）
    base_name = os.path.basename(npz_file_path).replace('.npz', '')
    save_path = os.path.join(save_dir, f"{base_name}_distribution.png")
    
    # 保存图片：dpi=300 保证清晰度，bbox_inches='tight' 防止边缘标题被裁切
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已成功保存至: {save_path}")

    # 如果你在服务器上跑，不需要弹窗，可以直接把下面这行注释掉
    plt.show()

    

    # 5. [进阶] 绘制联合分布图 (Pairplot)
    # 观察特征组合起来是否能更好地区分
    print("正在绘制特征散点相关性矩阵...")
    sns.pairplot(
        df_all, 
        vars=['f_start', 'f_goal', 'd_obs_norm'], # 选几个主要标量特征
        hue='Class', 
        plot_kws={'alpha': 0.6, 's': 15},
        palette=['#FF5722', '#03A9F4']
    )
    plt.suptitle("Feature Pairplot (Joint Distributions)", y=1.02)
    plt.show()


def visualize_feature_distributions_B(npz_file_path, save_dir='./output_plots'):
    print(f"正在加载数据: {npz_file_path}")
    data = np.load(npz_file_path)
    points = data['points']  # shape: [N', 7] (xyz, f_start, f_goal, d_obs_norm, prob_A)
    labels = data['labels']  # shape: [N', 2] (两套真值)

    # 💡 只保留真实存在的 4 个特征
    feature_names = ['f_start', 'f_goal', 'd_obs_norm', 'prob_A']
    
    # 1. 精准提取第 3, 4, 5 列，以及最后 1 列 (-1)
    features = points[:, [3, 4, 5, -1]]
    
    # 提取 Model B 的航路点真值 (第二个通道，索引为1)
    point_scores = labels[:, 1] 

    # 排除起终点 
    f_start_val = features[:, 0]  # f_start
    f_goal_val = features[:, 1]   # f_goal
    mid_mask = (f_start_val < 0.95) & (f_goal_val < 0.95)

    # 过滤出非起终点的中间点
    features = features[mid_mask]
    point_scores = point_scores[mid_mask]

    # 2. 划分正负样本
    wp_mask = point_scores > 0.8
    bg_mask = point_scores < 0.4  # 负样本使用 < 0.1

    wp_features = features[wp_mask]
    bg_features = features[bg_mask]

    print(f"提取到正样本(航路点附近): {len(wp_features)} 个")
    print(f"提取到负样本(背景点): {len(bg_features)} 个")

    if len(wp_features) == 0:
        print("⚠️ 该样本中没有得分 > 0.8 的非起终点数据，已跳过绘图。")
        return

    # 3. 构建 DataFrame
    df_wp = pd.DataFrame(wp_features, columns=feature_names)
    df_wp['Class'] = 'Waypoint (Score > 0.8)'

    df_bg = pd.DataFrame(bg_features, columns=feature_names)
    df_bg['Class'] = 'Background (Score < 0.4)'

    df_all = pd.concat([df_wp, df_bg], ignore_index=True)

    # 4. 绘制特征的一维概率密度分布 (画布改为 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        sns.histplot(
            data=df_all, 
            x=feat,               
            hue='Class',          
            stat="density",       
            common_norm=False,    
            kde=True,             
            alpha=0.4,            
            ax=axes[i],
            palette=['#FF5722', '#03A9F4'] 
        )
        
        axes[i].set_title(f'Distribution of {feat}')
        axes[i].set_xlabel("Feature Value")
        axes[i].set_ylabel("Density")

    plt.suptitle("Feature Distribution Analysis (Stage B Validation Data)", fontsize=16)
    plt.tight_layout()

    # ==========================================
    # 保存图片逻辑 
    # ==========================================
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(npz_file_path).replace('.npz', '')
    
    # 保存 2x2 密度图
    save_path = os.path.join(save_dir, f"{base_name}_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 密度分布图已成功保存至: {save_path}")
    plt.close(fig)

    # # 5. [进阶] 绘制联合分布图 (Pairplot)
    # print("正在绘制特征散点相关性矩阵...")
    # pairplot_fig = sns.pairplot(
    #     df_all, 
    #     vars=feature_names, # 直接使用这 4 个特征
    #     hue='Class', 
    #     plot_kws={'alpha': 0.6, 's': 15},
    #     palette=['#FF5722', '#03A9F4']
    # )
    # plt.suptitle(f"Feature Pairplot - {base_name}", y=1.02)
    
    # pairplot_save_path = os.path.join(save_dir, f"{base_name}_pairplot.png")
    # pairplot_fig.savefig(pairplot_save_path, dpi=300, bbox_inches='tight')
    # print(f"✅ 相关性矩阵图已成功保存至: {pairplot_save_path}")
    # plt.close()


def compute_asymmetry_and_visualize(npz_file_path, R=3.0, save_dir='./output_plots'):
    """
    计算双向空间不对称性 (Bidirectional Flow Asymmetry) 并可视化分布
    参数:
    - R: 局部点云的搜索半径 (根据你的场景尺度调整，建议走廊宽度的 1~2 倍)
    """
    print(f"\n正在加载数据: {os.path.basename(npz_file_path)}")
    data = np.load(npz_file_path)
    points = data['points']  # [N, 7]
    labels = data['labels']  # [N, 2]

    # 1. 拆解基础数据
    xyz = points[:, 0:3]          # 三维坐标
    f_start = points[:, 3]        # 起点特征
    f_goal = points[:, 4]         # 终点特征
    point_scores = labels[:, 1]   # 航路点真值 (通道 2)

    # 2. 💡 巧妙获取起终点坐标 (寻找 f_start 和 f_goal 的最大值位置)
    start_pos = xyz[np.argmax(f_start)]
    goal_pos = xyz[np.argmax(f_goal)]

    # 3. 计算所有点指向起终点的单位向量 (u_start, u_goal)
    v_start = start_pos - xyz
    v_goal = goal_pos - xyz
    
    # 防止除以 0，加上 1e-8
    u_start = v_start / (np.linalg.norm(v_start, axis=1, keepdims=True) + 1e-8)
    u_goal = v_goal / (np.linalg.norm(v_goal, axis=1, keepdims=True) + 1e-8)

    # 4. 💡 利用 KDTree 进行高效的半径邻域搜索
    print(f"构建 KDTree 并搜索半径 R={R} 内的局部点云...")
    tree = KDTree(xyz)
    neighbors_list = tree.query_ball_point(xyz, r=R)

    # 5. 计算不对称性特征 (F_asym)
    N_points = len(xyz)
    F_asym = np.zeros(N_points)
    epsilon = 1e-5

    for i in range(N_points):
        neighbors_idx = neighbors_list[i]
        
        # 如果周围没有其他点，不对称度为 0
        if len(neighbors_idx) <= 1:
            continue
            
        # 提取周围邻居的坐标，并计算当前点指向邻居的向量
        P_i = xyz[i]
        P_neighbors = xyz[neighbors_idx]
        v_neighbors = P_neighbors - P_i
        
        # 利用点积判断方向 (夹角 < 90度，即点积 > 0)
        dot_start = np.dot(v_neighbors, u_start[i])
        dot_goal = np.dot(v_neighbors, u_goal[i])
        
        N_start = np.sum(dot_start > 0)
        N_goal = np.sum(dot_goal > 0)
        
        # 计算不对称度公式
        F_asym[i] = abs(N_start - N_goal) / (N_start + N_goal + epsilon)

    # 6. 排除起终点附近的点
    mid_mask = (f_start < 0.95) & (f_goal < 0.95)
    F_asym_mid = F_asym[mid_mask]
    scores_mid = point_scores[mid_mask]

    # 7. 划分正负样本
    wp_mask = scores_mid > 0.8
    bg_mask = scores_mid < 0.4

    wp_features = F_asym_mid[wp_mask]
    bg_features = F_asym_mid[bg_mask]

    print(f"提取到正样本(航路点): {len(wp_features)} 个")
    print(f"提取到负样本(背景点): {len(bg_features)} 个")

    if len(wp_features) == 0:
        print("⚠️ 该样本中正样本数量不足，跳过绘图。")
        return

    # 8. 绘制并保存概率密度分布图
    df_wp = pd.DataFrame({'F_asym': wp_features, 'Class': 'Waypoint (Score > 0.8)'})
    df_bg = pd.DataFrame({'F_asym': bg_features, 'Class': 'Background (Score < 0.4)'})
    df_all = pd.concat([df_wp, df_bg], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_all, 
        x='F_asym',               
        hue='Class',          
        stat="density",       
        common_norm=False,    
        kde=True,             
        alpha=0.4,            
        palette=['#FF5722', '#03A9F4'] 
    )
    
    plt.title("Bidirectional Flow Asymmetry Distribution", fontsize=16)
    plt.xlabel(f"Asymmetry Ratio ($F_{{asym}}$) within R={R}", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(npz_file_path).replace('.npz', '')
    save_path = os.path.join(save_dir, f"{base_name}_F_asym_R{R}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 不对称性分布图已保存至: {save_path}")


def compute_orthogonal_tension_and_visualize(npz_file_path, R=3.0, save_dir='./output_plots'):
    """
    计算目标引力与局部阻碍的正交性 (Orthogonal Tension) 并可视化分布
    参数:
    - R: 局部点云的搜索半径 (建议与你的走廊宽度或机器人感知半径相当)
    """
    print(f"\n正在加载数据: {os.path.basename(npz_file_path)}")
    data = np.load(npz_file_path)
    points = data['points']  # [N, 7]
    labels = data['labels']  # [N, 2]

    # 1. 拆解基础数据
    xyz = points[:, 0:3]          # 三维坐标
    f_start = points[:, 3]        # 起点特征
    f_goal = points[:, 4]         # 终点特征
    point_scores = labels[:, 1]   # 航路点真值 (通道 2)

    # 2. 巧妙获取终点坐标 (寻找 f_goal 的最大值位置)
    goal_pos = xyz[np.argmax(f_goal)]

    # 3. 利用 KDTree 进行高效的半径邻域搜索
    print(f"构建 KDTree 并搜索半径 R={R} 内的局部点云...")
    tree = KDTree(xyz)
    neighbors_list = tree.query_ball_point(xyz, r=R)

    # 4. 计算正交拉力特征 (F_ortho)
    N_points = len(xyz)
    F_ortho = np.zeros(N_points)

    for i in range(N_points):
        neighbors_idx = neighbors_list[i]
        
        # 如果周围点太少，无法形成有效的质心，跳过
        if len(neighbors_idx) <= 1:
            continue
            
        P_i = xyz[i]
        
        # 💡 第一步：计算局部点云的质心 C_obs
        C_obs = np.mean(xyz[neighbors_idx], axis=0)
        
        # 💡 第二步：计算挤压力方向 v_obs 和 引力方向 v_goal
        v_obs = C_obs - P_i
        v_goal = goal_pos - P_i
        
        norm_obs = np.linalg.norm(v_obs)
        norm_goal = np.linalg.norm(v_goal)
        
        # 防止除以 0 (例如当前点就是质心，或刚好在终点上)
        if norm_obs < 1e-6 or norm_goal < 1e-6:
            continue
            
        # 💡 第三步：计算点积，求余弦值
        cos_theta = np.dot(v_obs, v_goal) / (norm_obs * norm_goal)
        
        # 防止浮点数精度误差导致 cos_theta 略微超出 [-1, 1] 范围，引起 sqrt 报错
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # 💡 第四步：计算正交拉力 (实际上就是求 |sin(theta)|)
        offset_weight = min(norm_obs / R, 1.0) 
        
        # 计算加权后的正交拉力
        raw_ortho = np.sqrt(1.0 - cos_theta**2)
        F_ortho[i] = raw_ortho * offset_weight

    # 5. 排除起终点附近的点 (避免接近终点时 v_goal 发散导致的奇异性)
    mid_mask = (f_start < 0.95) & (f_goal < 0.95)
    F_ortho_mid = F_ortho[mid_mask]
    scores_mid = point_scores[mid_mask]

    # 6. 划分正负样本
    wp_mask = scores_mid > 0.8
    bg_mask = scores_mid < 0.4

    wp_features = F_ortho_mid[wp_mask]
    bg_features = F_ortho_mid[bg_mask]

    print(f"提取到正样本(航路点): {len(wp_features)} 个")
    print(f"提取到负样本(背景点): {len(bg_features)} 个")

    if len(wp_features) == 0:
        print("⚠️ 该样本中正样本数量不足，跳过绘图。")
        return

    # 7. 绘制并保存概率密度分布图
    df_wp = pd.DataFrame({'F_ortho': wp_features, 'Class': 'Waypoint (Score > 0.8)'})
    df_bg = pd.DataFrame({'F_ortho': bg_features, 'Class': 'Background (Score < 0.4)'})
    df_all = pd.concat([df_wp, df_bg], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_all, 
        x='F_ortho',               
        hue='Class',          
        stat="density",       
        common_norm=False,    
        kde=True,             
        alpha=0.4,            
        palette=['#FF5722', '#03A9F4'] 
    )
    
    plt.title("Orthogonal Tension Distribution", fontsize=16)
    plt.xlabel(f"Orthogonal Tension ($F_{{ortho}}$) within R={R}", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(npz_file_path).replace('.npz', '')
    save_path = os.path.join(save_dir, f"{base_name}_F_ortho_R{R}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 正交拉力分布图已保存至: {save_path}")

def compute_flow_density_and_visualize(npz_file_path, R=3.0, save_dir='./output_plots'):
    """
    计算流场密度 (Flow Field Density) 并可视化分布
    参数:
    - R: 局部点云的搜索半径 (建议与之前正交拉力的半径保持一致，以便对比)
    """
    print(f"\n正在加载数据: {os.path.basename(npz_file_path)}")
    data = np.load(npz_file_path)
    points = data['points']  # [N, 7]
    labels = data['labels']  # [N, 2]

    # 1. 拆解基础数据
    xyz = points[:, 0:3]          # 三维坐标
    f_start = points[:, 3]        # 起点特征
    f_goal = points[:, 4]         # 终点特征
    point_scores = labels[:, 1]   # 航路点真值 (通道 2)

    # 2. 利用 KDTree 进行高效的半径邻域搜索
    print(f"构建 KDTree 并搜索半径 R={R} 内的局部点云...")
    tree = KDTree(xyz)
    # query_ball_point 返回的是一个列表，列表里每个元素是该点邻域内的点索引数组
    neighbors_list = tree.query_ball_point(xyz, r=R)

    # 3. 💡 计算每个点的流场密度 (即邻域内的点云数量)
    # 使用列表推导式高效获取每个邻域内的点数量
    local_point_counts = np.array([len(indices) for indices in neighbors_list], dtype=np.float32)

    # 4. 💡 归一化到 0-1 范围 (Min-Max Normalization)
    min_count = np.min(local_point_counts)
    max_count = np.max(local_point_counts)
    
    # 防止分母为 0 (极端情况下全图密度一样)
    if max_count - min_count < 1e-6:
        F_density = np.zeros_like(local_point_counts)
    else:
        F_density = (local_point_counts - min_count) / (max_count - min_count)

    # 5. 排除起终点附近的点
    mid_mask = (f_start < 0.95) & (f_goal < 0.95)
    F_density_mid = F_density[mid_mask]
    scores_mid = point_scores[mid_mask]

    # 6. 划分正负样本
    wp_mask = scores_mid > 0.8
    bg_mask = scores_mid < 0.4

    wp_features = F_density_mid[wp_mask]
    bg_features = F_density_mid[bg_mask]

    print(f"提取到正样本(航路点): {len(wp_features)} 个")
    print(f"提取到负样本(背景点): {len(bg_features)} 个")

    if len(wp_features) == 0:
        print("⚠️ 该样本中正样本数量不足，跳过绘图。")
        return

    # 7. 绘制并保存概率密度分布图
    df_wp = pd.DataFrame({'F_density': wp_features, 'Class': 'Waypoint (Score > 0.8)'})
    df_bg = pd.DataFrame({'F_density': bg_features, 'Class': 'Background (Score < 0.4)'})
    df_all = pd.concat([df_wp, df_bg], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_all, 
        x='F_density',               
        hue='Class',          
        stat="density",       
        common_norm=False,    
        kde=True,             
        alpha=0.4,            
        palette=['#FF5722', '#03A9F4'] 
    )
    
    plt.title("Flow Field Density Distribution", fontsize=16)
    plt.xlabel(f"Normalized Density ($F_{{density}}$) within R={R}", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(npz_file_path).replace('.npz', '')
    save_path = os.path.join(save_dir, f"{base_name}_F_density_R{R}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 流场密度分布图已保存至: {save_path}")

def compute_pca_anisotropy_and_visualize(npz_file_path, R=3.0, save_dir='./output_plots'):
    """
    计算局部空间各向异性 (PCA Shape Descriptors) 并可视化
    包含三个特征：线性度 (Linearity)、平面度 (Planarity)、散布度 (Scatter)
    """
    print(f"\n正在加载数据: {os.path.basename(npz_file_path)}")
    data = np.load(npz_file_path)
    points = data['points']  # [N, 7]
    labels = data['labels']  # [N, 2]

    # 1. 拆解基础数据
    xyz = points[:, 0:3]          # 三维坐标
    f_start = points[:, 3]        # 起点特征
    f_goal = points[:, 4]         # 终点特征
    point_scores = labels[:, 1]   # 航路点真值 (通道 2)

    # 2. KDTree 邻域搜索
    print(f"构建 KDTree 并搜索半径 R={R} 内的局部点云...")
    tree = KDTree(xyz)
    neighbors_list = tree.query_ball_point(xyz, r=R)

    # 3. 初始化三个形状特征
    N_points = len(xyz)
    F_line = np.zeros(N_points)
    F_plane = np.zeros(N_points)
    F_scatter = np.zeros(N_points)
    
    epsilon = 1e-8 # 防止除以 0

    print("正在进行局部 PCA 计算 (这可能需要几秒钟)...")
    for i in range(N_points):
        neighbors_idx = neighbors_list[i]
        
        # PCA 至少需要 3 个点才能计算 3D 协方差矩阵
        if len(neighbors_idx) < 3:
            continue
            
        P_neighbors = xyz[neighbors_idx]
        
        # 💡 PCA 核心计算
        # 计算协方差矩阵 (rowvar=False 表示每一列是一个特征 xyz)
        cov_matrix = np.cov(P_neighbors, rowvar=False)
        
        # 提取特征值。eigvalsh 专门用于对称矩阵，速度更快且稳定
        # 注意：eigvalsh 返回的特征值是【升序】排列的 (l3, l2, l1)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # 修正可能出现的极微小负数误差，并倒序排成【降序】 (l1 >= l2 >= l3)
        eigenvalues = np.clip(eigenvalues, 0, None)
        l1, l2, l3 = eigenvalues[::-1]
        
        # 如果最大特征值几乎为 0，说明所有点都重合了，跳过
        if l1 < epsilon:
            continue
            
        # 💡 计算形状描述子
        F_line[i] = (l1 - l2) / l1
        F_plane[i] = (l2 - l3) / l1
        F_scatter[i] = l3 / l1

    # 4. 排除起终点附近的点
    mid_mask = (f_start < 0.95) & (f_goal < 0.95)
    F_line_mid = F_line[mid_mask]
    F_plane_mid = F_plane[mid_mask]
    F_scatter_mid = F_scatter[mid_mask]
    scores_mid = point_scores[mid_mask]

    # 5. 划分正负样本
    wp_mask = scores_mid > 0.8
    bg_mask = scores_mid < 0.4

    print(f"提取到正样本(航路点): {np.sum(wp_mask)} 个")
    print(f"提取到负样本(背景点): {np.sum(bg_mask)} 个")

    if np.sum(wp_mask) == 0:
        print("⚠️ 该样本中正样本数量不足，跳过绘图。")
        return

    # 6. 构建 DataFrame 用于画图
    df_wp = pd.DataFrame({
        'Linearity': F_line_mid[wp_mask],
        'Planarity': F_plane_mid[wp_mask],
        'Scatter': F_scatter_mid[wp_mask],
        'Class': 'Waypoint (Score > 0.8)'
    })
    
    df_bg = pd.DataFrame({
        'Linearity': F_line_mid[bg_mask],
        'Planarity': F_plane_mid[bg_mask],
        'Scatter': F_scatter_mid[bg_mask],
        'Class': 'Background (Score < 0.4)'
    })
    
    df_all = pd.concat([df_wp, df_bg], ignore_index=True)

    # 7. 绘制 1x3 的密度分布图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    feature_names = ['Linearity', 'Planarity', 'Scatter']
    
    for i, feat in enumerate(feature_names):
        sns.histplot(
            data=df_all, 
            x=feat,               
            hue='Class',          
            stat="density",       
            common_norm=False,    
            kde=True,             
            alpha=0.4,            
            ax=axes[i],
            palette=['#FF5722', '#03A9F4'] 
        )
        axes[i].set_title(f'{feat} Distribution', fontsize=14)
        axes[i].set_xlabel(feat, fontsize=12)
        axes[i].set_ylabel("Density", fontsize=12)

    plt.suptitle(f"Local Spatial Anisotropy (PCA) within R={R}", fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(npz_file_path).replace('.npz', '')
    save_path = os.path.join(save_dir, f"{base_name}_PCA_Shape_R{R}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ PCA 形状特征分布图已保存至: {save_path}")

def append_geometric_features(input_npz_path, output_npz_path, R=0.1):
    """
    为原始 npz 文件计算并追加 3 个新的几何特征，并保存到新路径。
    原有 7 维特征：[x, y, z, f_start, f_goal, d_obs_norm, prob_A]
    追加 3 维特征：[F_ortho, F_density, F_scatter]
    最终特征维度：10 维
    """
    # 1. 加载数据
    data = dict(np.load(input_npz_path)) # 💡 将所有键值对转为字典
    points = data['points']  # [N, 7]
    labels = data['labels']  # [N, 2]
    
    xyz = points[:, 0:3]
    f_start = points[:, 3]
    f_goal = points[:, 4]
    N = len(xyz)
    
    # 2. 获取终点坐标 (寻找 f_goal 的最大值位置)
    goal_pos = xyz[np.argmax(f_goal)]
    
    # 3. 统一构建 KDTree 并搜索邻域
    tree = KDTree(xyz)
    neighbors_list = tree.query_ball_point(xyz, r=R)
    
    # 4. 初始化三个新特征数组
    F_ortho = np.zeros(N)
    F_scatter = np.zeros(N)
    
    # --- 计算 1: 流场密度 (Flow Field Density) ---
    local_counts = np.array([len(idx) for idx in neighbors_list], dtype=np.float32)
    min_c, max_c = np.min(local_counts), np.max(local_counts)
    if max_c - min_c < 1e-6:
        F_density = np.zeros(N)
    else:
        F_density = (local_counts - min_c) / (max_c - min_c)
        
    # 遍历每个点计算另外两个特征
    for i in range(N):
        neighbors_idx = neighbors_list[i]
        
        # 至少需要 3 个点才能进行 PCA 和质心计算
        if len(neighbors_idx) < 3:
            continue
            
        P_i = xyz[i]
        P_neighbors = xyz[neighbors_idx]
        
        # --- 计算 2: 正交拉力 (Orthogonal Tension - Weighted) ---
        C_obs = np.mean(P_neighbors, axis=0)
        v_obs = C_obs - P_i
        v_goal = goal_pos - P_i
        
        norm_obs = np.linalg.norm(v_obs)
        norm_goal = np.linalg.norm(v_goal)
        
        if norm_obs >= 1e-6 and norm_goal >= 1e-6:
            cos_theta = np.dot(v_obs, v_goal) / (norm_obs * norm_goal)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            offset_weight = min(norm_obs / R, 1.0)
            F_ortho[i] = np.sqrt(1.0 - cos_theta**2) * offset_weight
            
        # --- 计算 3: 形状散布度 (PCA Scatter) ---
        cov_matrix = np.cov(P_neighbors, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.clip(eigenvalues, 0, None) # 防止精度误差产生的负数
        l1, l2, l3 = eigenvalues[::-1]              # 降序排列
        
        if l1 >= 1e-8:
            F_scatter[i] = l3 / l1
            
    # 5. 💡 起终点置零掩码 (Zero-Masking)
    # 起终点附近的局部几何极其不稳定且无意义，强制抹平为 0
    mask_near_ends = (f_start >= 0.95) | (f_goal >= 0.95)
    # F_ortho[mask_near_ends] = 0.0
    # F_density[mask_near_ends] = 0.0
    # F_scatter[mask_near_ends] = 0.0
    
    # 6. 维度拓展与拼接
    # 将形状变为 [N, 1] 以便拼接
    F_ortho = F_ortho.reshape(-1, 1)
    F_density = F_density.reshape(-1, 1)
    F_scatter = F_scatter.reshape(-1, 1)
    
    # 最终拼接: points 从 [N, 7] 变成 [N, 10]
    new_points = np.concatenate([points, F_ortho, F_density, F_scatter], axis=1)
    
    # 7. 💡 覆盖字典里的 points，保留其他所有原有数据 (比如 waypoints)
    data['points'] = new_points
    
    # 8. 保存到新路径 (使用 ** 解包字典，保存所有键值对)
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez_compressed(output_npz_path, **data)

def visualize_enriched_feature_distributions(npz_file_path, save_dir='./output_plots'):
    """
    针对特征增强后的 10 维数据，绘制 7 个标量特征的概率密度分布图。
    """
    print(f"正在加载增强数据: {os.path.basename(npz_file_path)}")
    data = np.load(npz_file_path)
    points = data['points']  # shape: [N, 10] 
    labels = data['labels']  # shape: [N, 2]

    # 💡 更新为 7 个标量特征的名称
    feature_names = [
        'f_start', 'f_goal', 'd_obs_norm', 'prob_A', 
        'F_ortho', 'F_density', 'F_scatter'
    ]
    
    # 1. 精准提取第 3 到第 9 列 (一共 7 列)
    features = points[:, 3:10]
    
    # 提取 Model B 的航路点真值 (第二个通道，索引为1)
    point_scores = labels[:, 1] 

    # 排除起终点 (在截取后的 features 数组中，f_start 和 f_goal 的索引变成了 0 和 1)
    f_start_val = features[:, 0]  
    f_goal_val = features[:, 1]   
    mid_mask = (f_start_val < 0.95) & (f_goal_val < 0.95)

    # 过滤出非起终点的中间点
    features = features[mid_mask]
    point_scores = point_scores[mid_mask]

    # 2. 划分正负样本
    wp_mask = point_scores > 0.8
    bg_mask = point_scores < 0.4  

    wp_features = features[wp_mask]
    bg_features = features[bg_mask]

    print(f"提取到正样本(航路点附近): {len(wp_features)} 个")
    print(f"提取到负样本(背景点): {len(bg_features)} 个")

    if len(wp_features) == 0:
        print("⚠️ 该样本中没有得分 > 0.8 的非起终点数据，已跳过绘图。")
        return

    # 3. 构建 DataFrame
    df_wp = pd.DataFrame(wp_features, columns=feature_names)
    df_wp['Class'] = 'Waypoint (Score > 0.8)'

    df_bg = pd.DataFrame(bg_features, columns=feature_names)
    df_bg['Class'] = 'Background (Score < 0.4)'

    df_all = pd.concat([df_wp, df_bg], ignore_index=True)

    # 4. 💡 绘制特征的一维概率密度分布 (画布改为 2行4列 = 8个图表位)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        sns.histplot(
            data=df_all, 
            x=feat,               
            hue='Class',          
            stat="density",       
            common_norm=False,    
            kde=True,             
            alpha=0.4,            
            ax=axes[i],
            palette=['#FF5722', '#03A9F4'] 
        )
        
        axes[i].set_title(f'Distribution of {feat}')
        axes[i].set_xlabel("Feature Value")
        axes[i].set_ylabel("Density")

    # 💡 隐藏第 8 个多余的空白子图 (因为我们只有 7 个特征)
    axes[7].set_visible(False)

    plt.suptitle("Enriched Feature Distribution Analysis (Stage B Data)", fontsize=18)
    plt.tight_layout()

    # ==========================================
    # 保存图片逻辑 
    # ==========================================
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(npz_file_path).replace('.npz', '')
    
    # 保存 2x4 密度图
    save_path = os.path.join(save_dir, f"{base_name}_enriched_dist.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 10维特征密度分布图已成功保存至: {save_path}")
    plt.close(fig)

def analyze_single_label_distribution(npz_file_path, target_channel=2, save_dir='./output_plots'):
    """
    精准剖析 npz 文件中【指定通道】的标签数值分布情况。
    参数:
    - target_channel: 想要查看的通道编号 (1 代表第一个通道，2 代表第二个通道)
    """
    print(f"\n" + "="*50)
    print(f"📊 正在分析数据: {os.path.basename(npz_file_path)} | 目标通道: {target_channel}")
    print("="*50)
    
    # 1. 加载数据
    data = np.load(npz_file_path)
    if 'labels' not in data:
        print("❌ 错误：文件中未找到 'labels' 键！")
        return
        
    labels = data['labels']  
    
    # 2. 鲁棒的维度处理
    # 如果 labels 是一维的 [N]，强行转为 [N, 1] 以便统一处理
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
        
    max_channels = labels.shape[1]
    channel_idx = target_channel - 1  # 索引从 0 开始
    
    if channel_idx < 0 or channel_idx >= max_channels:
        print(f"❌ 错误：请求的通道 {target_channel} 超出范围！该文件仅有 {max_channels} 个标签通道。")
        return

    # 提取目标通道数据
    ch_data = labels[:, channel_idx]
    total_points = len(ch_data)

    # 3. 核心统计信息计算
    print(f"\n>>> 通道 {target_channel} 统计信息 <<<")
    print(f"  • 总点数: {total_points}")
    print(f"  • 最小值: {np.min(ch_data):.4f} | 最大值: {np.max(ch_data):.4f} | 平均值: {np.mean(ch_data):.4f}")
    
    # 统计不同分数段的分布
    num_zeros = np.sum(ch_data == 0)
    num_ones = np.sum(ch_data == 1)
    num_neg = np.sum(ch_data <= 0.1)
    num_pos = np.sum(ch_data >= 0.8)
    num_mid = np.sum((ch_data > 0.1) & (ch_data < 0.8))
    
    print(f"  • 绝对 0 的点数: {num_zeros} ({num_zeros/total_points*100:.2f}%)")
    print(f"  • 绝对 1 的点数: {num_ones} ({num_ones/total_points*100:.2f}%)")
    print(f"  • 背景点 (<= 0.1): {num_neg} ({num_neg/total_points*100:.2f}%)")
    print(f"  • 航路点 (>= 0.8): {num_pos} ({num_pos/total_points*100:.2f}%)")
    print(f"  • 模棱两可点 (0.1 ~ 0.8): {num_mid} ({num_mid/total_points*100:.2f}%)")
    
    # 计算正负样本不平衡比
    if num_pos > 0:
        imbalance_ratio = num_neg / num_pos
        print(f"  🚨 正负样本不平衡比例 (<=0.1 vs >=0.8) ≈ 1 : {imbalance_ratio:.1f}")
    else:
        print("  🚨 警告: 未检测到得分 >= 0.8 的正样本！")

    # 4. 绘制单通道概率密度直方图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 动态选择颜色
    color_map = {1: '#FF5722', 2: '#03A9F4'}
    plot_color = color_map.get(target_channel, '#4CAF50')

    # ---------------------------------------------
    # 图 1：线性频率分布 (Relative Frequency)
    # 纵坐标是 0 到 1 之间的小数，代表该柱子内的点占总数的比例
    # (注意：因为极度不平衡，高分区的柱子在这里可能几乎看不见)
    # ---------------------------------------------
    sns.histplot(
        ch_data, 
        bins=50, 
        kde=False,  
        stat='probability',  # 💡 核心修改：指定为频率 (概率比例)
        color=plot_color, 
        ax=axes[0]
    )
    axes[0].set_title(f"Label Frequency (Linear Scale)", fontsize=14)
    axes[0].set_xlabel("Label Value (Score)", fontsize=12)
    axes[0].set_ylabel("Frequency (Proportion)", fontsize=12)

    # ---------------------------------------------
    # 图 2：对数频率分布 (Log Frequency)
    # 纵坐标依然是频率比例，但使用对数轴，让你能看清那些只占 0.01% 的正样本
    # ---------------------------------------------
    sns.histplot(
        ch_data, 
        bins=50, 
        kde=False, 
        stat='probability',  # 💡 同样指定为频率
        color=plot_color, 
        ax=axes[1]
    )
    axes[1].set_yscale('log')  # 开启对数放大镜
    axes[1].set_title(f"Label Frequency (Log Scale - Zoom in)", fontsize=14)
    axes[1].set_xlabel("Label Value (Score)", fontsize=12)
    axes[1].set_ylabel("Frequency (Log Scale)", fontsize=12)

    plt.suptitle(f"Label Frequency Analysis - Channel {target_channel}\n({os.path.basename(npz_file_path)})", fontsize=16, y=1.05)
    plt.tight_layout()

    # 5. 保存图片
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(npz_file_path).replace('.npz', '')
    save_path = os.path.join(save_dir, f"{base_name}_label_Ch{target_channel}_dist.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 标签分布直方图已保存至: {save_path}")
    print("="*50)

# 调用测试
if __name__ == "__main__":
    # # 基础路径配置
    # base_dir = r"C:\Users\Administrator\Desktop\experiments\train_data16"
    # save_directory = os.path.join(base_dir, "plots")  # 保存图片的文件夹路径
    
    # # 你想随机抽取测试的文件数量
    # num_test_samples = 5 
    
    # print(f"🚀 开始批量随机测试，计划抽取 {num_test_samples} 个文件...")
    
    # for i in range(num_test_samples):
    #     # 💡 随机生成 map 和 task 的数字
    #     map_id = random.randint(0, 0)     # 生成 0 到 49 之间的随机整数
    #     task_id = random.randint(0, 9)   # 生成 0 到 19 之间的随机整数
        
    #     # 拼接出完整的文件名和路径
    #     file_name = f"map{map_id}_task{task_id}.npz"
    #     sample_file = os.path.join(base_dir, file_name)
        
    #     print("-" * 40)
    #     print(f"[{i+1}/{num_test_samples}] 尝试处理: {file_name}")
        
    #     if os.path.exists(sample_file):
    #         visualize_feature_distributions(sample_file, save_directory)
    #     else:
    #         print(f"❌ 未找到文件: {sample_file}，已跳过。")
            
    # print("-" * 40)
    # print("✅ 批量测试结束！请前往 plots 文件夹查看生成的图片。")
    # 💡 指向新的验证集目录
    # 指向新的验证集目录
    # val_dir = r"C:\Users\Administrator\Desktop\experiments\train_data16\val"
    # save_directory = os.path.join(val_dir, "plots")  # 保存图片的文件夹路径
    
    # # 获取所有的 .npz 文件
    # all_val_files = glob.glob(os.path.join(val_dir, "*.npz"))
    
    # if len(all_val_files) == 0:
    #     print(f"❌ 错误: 在 {val_dir} 下没有找到任何 .npz 文件！")
    #     exit()

    # num_test_samples = 20 
    
    # # 随机抽取真实存在的文件
    # test_files = random.sample(all_val_files, min(num_test_samples, len(all_val_files)))
    
    # print(f"🚀 开始批量随机测试，计划抽取 {len(test_files)} 个文件...")
    
    # for i, sample_file in enumerate(test_files):
    #     print("-" * 50)
    #     print(f"[{i+1}/{len(test_files)}] 尝试处理: {os.path.basename(sample_file)}")
    #     # 注意：这里调用的是改名后的 visualize_feature_distributions2
    #     visualize_feature_distributions_B(sample_file, save_directory)
            
    # print("-" * 50)
    # print(f"✅ 批量测试结束！请前往 {save_directory} 查看生成的图片。")

    # SEARCH_RADIUS = 0.1
    
    # # 随机抽样测试
    # test_files = random.sample(all_val_files, min(20, len(all_val_files)))
    
    # # for file in test_files:
    #     # compute_asymmetry_and_visualize(file, R=SEARCH_RADIUS, save_dir=save_directory)
    #     # compute_orthogonal_tension_and_visualize(file, R=SEARCH_RADIUS, save_dir=save_directory)
    #     # compute_flow_density_and_visualize(file, R=SEARCH_RADIUS, save_dir=save_directory)
    #     # compute_pca_anisotropy_and_visualize(file, R=SEARCH_RADIUS, save_dir=save_directory)
        
    # print("-" * 50)
    # print(f"✅ 特征检验完成！请前往 {save_directory} 查看。")


    # # ------------------------------生成新数据--------------------------------
    # # 1. 配置你的基础路径
    # base_data_dir = r"C:\Users\Administrator\Desktop\experiments\train_data16"
    
    # # 2. 定义你要提取的源文件夹字典 {"前缀名": "真实路径"}
    # input_dirs = {
    #     "train": os.path.join(base_data_dir, "train"),
    #     "val": os.path.join(base_data_dir, "val")
    # }
    
    # # 3. 定义统一的最终输出文件夹 (所有的 10 维数据都会汇聚于此)
    # output_dir = os.path.join(base_data_dir, "enriched_all_stage_B")
    # os.makedirs(output_dir, exist_ok=True)
    
    # # 💡 核心超参
    # SEARCH_RADIUS = 0.1
    
    # total_processed = 0
    
    # # 4. 开始遍历不同的数据集 (train, val)
    # for split_name, in_dir in input_dirs.items():
    #     all_files = glob.glob(os.path.join(in_dir, "*.npz"))
        
    #     if len(all_files) == 0:
    #         print(f"⚠️ 警告: 未在 {in_dir} 找到文件，跳过...")
    #         continue
            
    #     print(f"\n🚀 开始提取 [{split_name}] 集数据，共发现 {len(all_files)} 个文件...")
        
    #     # 使用 tqdm 显示进度条
    #     for file_path in tqdm(all_files, desc=f"Processing {split_name}"):
    #         original_name = os.path.basename(file_path)
            
    #         # 💡 关键防覆盖设计：加上前缀，变成类似 "train_stage_b_data_00000.npz"
    #         new_name = f"{split_name}_{original_name}" 
    #         save_path = os.path.join(output_dir, new_name)
            
    #         # 调用核心提取函数 (核心函数 append_geometric_features 保持不变)
    #         append_geometric_features(
    #             input_npz_path=file_path, 
    #             output_npz_path=save_path, 
    #             R=SEARCH_RADIUS
    #         )
    #         total_processed += 1
            
    # print(f"\n" + "="*50)
    # print(f"✅ 大功告成！合并数据处理完毕！")
    # print(f"📊 总计生成了 {total_processed} 个增强样本 (10 维特征)。")
    # print(f"📦 所有数据已安全汇聚至: {output_dir}")
    # print("="*50)

    # -------------------------------验证新数据的分布--------------------------------
    # enriched_data_dir = r"C:\Users\Administrator\Desktop\experiments\train_data16\enriched_all_stage_B"
    # save_directory = os.path.join(enriched_data_dir, "plots_check") 
    
    # # 获取所有的 .npz 文件
    # all_files = glob.glob(os.path.join(enriched_data_dir, "*.npz"))
    
    # if len(all_files) == 0:
    #     print(f"❌ 错误: 在 {enriched_data_dir} 下没有找到任何文件！请确保增强脚本已成功运行。")
    #     exit()

    # # 随机抽取 5 个文件看看效果
    # test_files = random.sample(all_files, min(20, len(all_files)))
    
    # print(f"🚀 开始随机抽检 {len(test_files)} 个增强版样本...")
    # for i, sample_file in enumerate(test_files):
    #     print("-" * 50)
    #     print(f"[{i+1}/{len(test_files)}] 分析文件: {os.path.basename(sample_file)}")
    #     visualize_enriched_feature_distributions(sample_file, save_directory)
            
    # print("-" * 50)
    # print(f"✅ 增强特征检验完成！请前往 {save_directory} 查看合并后的特征大图。")

    # -------------------------------单通道标签分布分析--------------------------------
    data_dir = r"C:\Users\Administrator\Desktop\experiments\train_data16\enriched_all_stage_B"
    save_directory = os.path.join(data_dir, "plots_labels") 
    
    # 💡 核心参数：你想看第几个通道？
    # Model A 数据通常看 1，Model B 数据通常看 2
    TARGET_CH = 2 
    
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if len(all_files) == 0:
        print(f"❌ 错误: 未在 {data_dir} 找到 npz 文件！")
    else:
        # 随机抽取 1 个文件查看
        sample_file = all_files[0] 
        analyze_single_label_distribution(sample_file, target_channel=TARGET_CH, save_dir=save_directory)

    