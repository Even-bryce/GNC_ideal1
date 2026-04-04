import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from res_show_for_data import plot_map
import random

def env_generator(
    rho=0.8,
    map_dim=(1500, 1500, 300),   # (Lx, Ly, Lz)
    r_crash_range=(30, 50),      # 碰撞半径
    r_risk_range=(3, 7),         # 风险半径偏移量（基于 r_crash）
    zmax_range=(30, 240),
    max_iter=5000,
    seed=None,
):
    """
    生成随机圆柱障碍物地图（允许交叉）
    r_risk = r_crash + 随机偏移量
    """

    if seed is not None:
        np.random.seed(seed)

    Lx, Ly, Lz = map_dim

    obstacle_list = []
    map_area = Lx * Ly

    r_avg = (r_crash_range[0] + r_crash_range[1]) / 2
    obs_area_avg = np.pi * r_avg ** 2
    num_obs_est = int(rho * map_area / obs_area_avg)

    for _ in range(num_obs_est):
        for _ in range(max_iter):

            # ===== 半径 =====
            r_crash = np.random.uniform(*r_crash_range)
            r_risk  = r_crash + np.random.uniform(*r_risk_range)

            # ===== 平面位置（保证风险圈不越界）=====
            x = np.random.uniform(r_risk, Lx - r_risk)
            y = np.random.uniform(r_risk, Ly - r_risk)

            # ===== 高度 =====
            zmin = 0.0
            zmax = np.random.uniform(zmax_range[0], min(zmax_range[1], Lz))

            obstacle_list.append((x, y, zmin, zmax, r_crash, r_risk))
            break

        else:
            print("Warning: reached max_iter, some obstacles not placed.")
            break

    map_dict = {
        "map_dim": map_dim,          #  核心：统一空间尺度
        "rho": rho,
        "area": map_area,
        "obstacles": obstacle_list,
        "num_obstacles": len(obstacle_list),
        "seed": seed,
    }

    return map_dict


def env_generator_maze(
    grid_size=(4, 4),            # 新增：迷宫的网格大小 (nx, ny)
    map_dim=(1500, 1500, 300),   # (Lx, Ly, Lz)
    r_crash_range=(40, 60),      # 碰撞半径 (适当调小以保证通道足够宽)
    r_risk_range=(5, 10),        # 风险半径偏移量
    zmax_range=(30, 240),
    seed=None,
):
    """
    生成由圆柱体连成线的曲折迷宫地图
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    Lx, Ly, Lz = map_dim
    nx, ny = grid_size
    
    # 计算每个网格（通道）的物理尺寸
    cell_w = Lx / nx
    cell_h = Ly / ny

    # ==========================================
    # 1. 使用 DFS 算法生成迷宫逻辑结构
    # ==========================================
    visited = np.zeros((nx, ny), dtype=bool)
    # h_walls[i, j] 表示第 i 列，第 j 行下方的水平墙
    h_walls = np.ones((nx, ny + 1), dtype=bool) 
    # v_walls[i, j] 表示第 i 列左侧，第 j 行的垂直墙
    v_walls = np.ones((nx + 1, ny), dtype=bool) 

    def dfs(x, y):
        visited[x, y] = True
        # 上右下左四个方向
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions) # 随机打乱方向，产生曲折迷宫

        for dx, dy in directions:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny and not visited[nx_, ny_]:
                # 打通相邻网格之间的墙壁
                if dx == 1: v_walls[nx_, y] = False
                if dx == -1: v_walls[x, y] = False
                if dy == 1: h_walls[x, ny_] = False
                if dy == -1: h_walls[x, y] = False
                dfs(nx_, ny_)

    # 从左下角 (0,0) 开始生成迷宫
    dfs(0, 0)

    # 挖去左下角和右上角的墙壁，作为整个地图的起点和终点出入口（可选）
    v_walls[0, 0] = False
    v_walls[nx, ny-1] = False

    # ==========================================
    # 2. 将逻辑墙壁转化为连续重叠的圆柱体
    # ==========================================
    obstacle_list = []
    
    # 边缘留白，防止圆柱体的风险圈超出边界报错
    pad = r_crash_range[1] + r_risk_range[1] 

    def place_wall_segment(x1, y1, x2, y2):
        """沿给定的线段密集放置圆柱体"""
        L = np.hypot(x2 - x1, y2 - y1)
        if L <= 0.01: return
        
        # 使用基础半径决定步长，步长等于 1.2 倍半径（确保紧密交叉连成线）
        r_base = (r_crash_range[0] + r_crash_range[1]) / 2
        step = r_base * 1.2 
        num_steps = max(2, int(np.ceil(L / step)) + 1)

        for i in range(num_steps):
            # 线性插值计算圆柱体圆心
            t = i / (num_steps - 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            # 生成单个圆柱体属性
            r_crash = np.random.uniform(*r_crash_range)
            r_risk = r_crash + np.random.uniform(*r_risk_range)
            zmax = np.random.uniform(zmax_range[0], min(zmax_range[1], Lz))

            obstacle_list.append((x, y, 0.0, zmax, r_crash, r_risk))

    # 遍历所有存在的水平墙
    for i in range(nx):
        for j in range(ny + 1):
            if h_walls[i, j]:
                x1 = max(pad, i * cell_w)
                x2 = min(Lx - pad, (i + 1) * cell_w)
                y_pos = np.clip(j * cell_h, pad, Ly - pad)
                place_wall_segment(x1, y_pos, x2, y_pos)

    # 遍历所有存在的垂直墙
    for i in range(nx + 1):
        for j in range(ny):
            if v_walls[i, j]:
                x_pos = np.clip(i * cell_w, pad, Lx - pad)
                y1 = max(pad, j * cell_h)
                y2 = min(Ly - pad, (j + 1) * cell_h)
                place_wall_segment(x_pos, y1, x_pos, y2)

    map_dict = {
        "map_dim": map_dim,
        "grid_size": grid_size,
        "area": Lx * Ly,
        "obstacles": obstacle_list,
        "num_obstacles": len(obstacle_list),
        "seed": seed,
    }

    return map_dict

def env_generator_cluster(
    map_dim=(1500, 1500, 240),   # (Lx, Ly, Lz)
    num_clusters=15,             # 建议 10~15 之间，保证有足够空间
    chain_length_range=(1, 4),   # 每个簇的圆柱体数量
    r_center_range=(70, 120),    # 接近地图中心的圆柱体半径范围
    r_edge_range=(20, 50),       # 接近地图边缘的圆柱体半径范围
    r_risk_range=(10, 20),       # 风险半径偏移量
    zmax_range=(30, 240),
    min_center_dist=400,         # 【核心参数】任意两个簇中心点的最小绝对距离！
    seed=None,
):
    """
    先严格生成保持距离的簇中心，再根据中心位置动态决定半径生成圆柱簇
    """
    if seed is not None:
        np.random.seed(seed)

    Lx, Ly, Lz = map_dim
    obstacle_list = []
    
    margin = r_edge_range[1] + r_risk_range[1]
    map_center_x, map_center_y = Lx / 2, Ly / 2
    max_dist_to_center = np.hypot(map_center_x, map_center_y)

    # ==========================================
    # 第一步：严格生成有间距的“簇中心点”
    # ==========================================
    cluster_centers = []
    max_attempts = 2000  # 尝试寻找合法位置的最大次数
    
    for _ in range(num_clusters):
        for _ in range(max_attempts):
            cx = np.random.uniform(margin, Lx - margin)
            cy = np.random.uniform(margin, Ly - margin)
            
            # 如果是第一个点，直接收下
            if not cluster_centers:
                cluster_centers.append((cx, cy))
                break
                
            # 计算当前点与所有已存在中心点的距离
            dists = [np.hypot(cx - px, cy - py) for px, py in cluster_centers]
            
            # 【核心逻辑】：只有当离所有其他中心都足够远时，才接受这个点
            if min(dists) >= min_center_dist:
                cluster_centers.append((cx, cy))
                break

    if len(cluster_centers) < num_clusters:
        print(f"提示：由于 min_center_dist ({min_center_dist}) 限制过大，在地图空间内只成功放置了 {len(cluster_centers)} 个簇。")

    # ==========================================
    # 第二步：根据中心点生成圆柱簇，动态调整半径
    # ==========================================
    for cx, cy in cluster_centers:
        
        # 计算该中心点距离地图核心的远近 (0 为绝对中心，1 为绝对边缘)
        dist_to_map_center = np.hypot(cx - map_center_x, cy - map_center_y)
        dist_ratio = min(1.0, dist_to_map_center / max_dist_to_center)
        
        # 动态插值计算当前这个簇应该使用的半径范围
        # （中心大，边缘小）
        curr_r_min = r_center_range[0] * (1 - dist_ratio) + r_edge_range[0] * dist_ratio
        curr_r_max = r_center_range[1] * (1 - dist_ratio) + r_edge_range[1] * dist_ratio

        # 决定该簇内部包含几个圆柱
        chain_length = np.random.randint(chain_length_range[0], chain_length_range[1] + 1)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # 游标从中心点开始
        curr_x, curr_y = cx, cy

        for _ in range(chain_length):
            # 使用专属的半径范围生成圆柱
            r_crash = np.random.uniform(curr_r_min, curr_r_max)
            r_risk = r_crash + np.random.uniform(*r_risk_range)
            zmax = np.random.uniform(zmax_range[0], min(zmax_range[1], Lz))
            
            # 边界保护
            if r_risk <= curr_x <= Lx - r_risk and r_risk <= curr_y <= Ly - r_risk:
                obstacle_list.append((curr_x, curr_y, 0.0, zmax, r_crash, r_risk))
            else:
                break # 一旦某个圆柱的圈会碰壁，直接打断这条线的延伸
            # 计算簇内下一个圆柱的位置 (使其紧密连接)
            step_size = r_crash * np.random.uniform(0.7, 1.0)
            
            # 控制延伸方向
            turn_prob = np.random.rand()
            if turn_prob < 0.6:
                angle += np.random.uniform(-np.pi/4, np.pi/4) 
            elif turn_prob < 0.9:
                angle += np.random.choice([-np.pi/2, np.pi/2])
            else:
                angle += np.random.uniform(-np.pi, np.pi)
                
            curr_x += step_size * np.cos(angle)
            curr_y += step_size * np.sin(angle)

    map_dict = {
        "map_dim": map_dim,
        "area": Lx * Ly,
        "obstacles": obstacle_list,
        "num_obstacles": len(obstacle_list),
        "seed": seed,
    }

    return map_dict

# -------测试-------
if __name__ == "__main__":

    # env_map = env_generator(
    #     rho=0.5,
    #     map_dim=(1500, 1500, 240),
    #     r_crash_range=(80, 125),
    #     r_risk_range=(15, 30),
    #     zmax_range=(240, 240),
    #     max_iter=5000,
    #     seed=40
    # )

    # env_map = env_generator_maze(
    #     grid_size=(4, 4),           # 4x4的网格，网格越多通道越窄越复杂
    #     map_dim=(1500, 1500, 240),
    #     r_crash_range=(40, 60),     # 为了给通道留出足够空间，半径相较于你原来设定的(80,125)稍微缩小了一些
    #     r_risk_range=(10, 20),
    #     zmax_range=(240, 240),
    #     seed=42
    # )

    env_map = env_generator_cluster(
    map_dim=(1500, 1500, 240),   # (Lx, Ly, Lz)
    num_clusters=20,             # 建议 10~15 之间，保证有足够空间
    chain_length_range=(1, 4),   # 每个簇的圆柱体数量
    r_center_range=(100, 150),    # 接近地图中心的圆柱体半径范围
    r_edge_range=(20, 50),       # 接近地图边缘的圆柱体半径范围
    r_risk_range=(10, 20),       # 风险半径偏移量
    zmax_range=(240, 240),
    min_center_dist=200,         # 【核心参数】任意两个簇中心点的最小绝对距离！
    seed=None,
)

    plot_map(env_map)
