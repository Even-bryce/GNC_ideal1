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


def env_generator_clutter(
    rho=0.8,
    map_dim=(1500, 1500, 300),   # (Lx, Ly, Lz)
    r_crash_range=(30, 50),      # 碰撞半径
    r_risk_range=(3, 7),         # 风险半径偏移量（基于 r_crash）
    zmax_range=(30, 240),
    max_iter=5000,               # 每次尝试摆放单个圆柱的最大失败次数
    seed=None,
):
    """
    生成随机杂乱圆柱障碍物地图（不允许圆柱重合）
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

    for i in range(num_obs_est):
        placed = False
        
        for _ in range(max_iter):
            # ===== 半径 =====
            r_crash = np.random.uniform(*r_crash_range)
            r_risk  = r_crash + np.random.uniform(*r_risk_range)

            # ===== 平面位置（保证风险圈不越界）=====
            x = np.random.uniform(r_risk, Lx - r_risk)
            y = np.random.uniform(r_risk, Ly - r_risk)

            # ==========================================
            # 核心修改：平面重叠检测
            # ==========================================
            is_overlapping = False
            for obs in obstacle_list:
                ox, oy, ozmin, ozmax, or_crash, or_risk = obs
                # 计算两圆心的平面欧氏距离
                dist = np.hypot(x - ox, y - oy) 
                
                # 如果距离小于两者物理碰撞半径之和，说明重合
                # （如果你希望连风险圈都不允许重合，可以将 r_crash 替换为 r_risk）
                if dist < (r_crash + or_crash): 
                    is_overlapping = True
                    break  # 发现重合，立刻跳出内层检测循环

            # 如果重合了，直接 continue 进入下一次 _ (max_iter) 重新随机位置
            if is_overlapping:
                continue 
            # ==========================================

            # ===== 高度 (如果没有重合，则走到这里) =====
            zmin = 0.0
            zmax = np.random.uniform(zmax_range[0], min(zmax_range[1], Lz))

            obstacle_list.append((x, y, zmin, zmax, r_crash, r_risk))
            placed = True
            break  # 当前障碍物摆放成功，跳出 max_iter 循环，去摆放下一个

        # 如果尝试了 max_iter 次依然 placed == False，说明地图实在太挤了放不下了
        if not placed:
            print(f"Warning: 地图过于拥挤！已达最大尝试次数 ({max_iter})。")
            print(f"实际生成障碍物: {len(obstacle_list)} / 预估: {num_obs_est}。可尝试降低 rho。")
            break  # 停止生成剩余的障碍物

    map_dict = {
        "map_dim": map_dim,          # 核心：统一空间尺度
        "rho": rho,
        "area": map_area,
        "obstacles": obstacle_list,
        "num_obstacles": len(obstacle_list),
        "seed": seed,
    }

    return map_dict

import numpy as np
import random

# ==========================================
# 辅助函数：计算点到【折线】的距离
# ==========================================
def is_point_near_polyline(p, waypoints, threshold):
    """检查点 p 是否在由多段线 (waypoints) 组成的路径的阈值距离内"""
    def point_near_segment(p, s1, s2, thresh):
        if np.all(s1 == s2): return np.linalg.norm(p - s1) <= thresh
        line_vec = s2 - s1
        point_vec = p - s1
        line_len_sq = np.sum(line_vec**2)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = s1 + t * line_vec
        return np.linalg.norm(p - projection) <= thresh

    for i in range(len(waypoints) - 1):
        if point_near_segment(p, waypoints[i], waypoints[i+1], threshold):
            return True
    return False

# ==========================================
# 辅助函数：生成横平竖直的连续圆柱墙 (带方向互斥逻辑)
# ==========================================
def add_uniform_cylinder_wall_safe(
    obstacle_list, existing_H, existing_V,  # 新增：传入已存在的横/纵坐标库
    Lx, Ly, Lz, cx, cy, direction, length, 
    zmax_range, r_crash, r_risk_offset, 
    safe_waypoints, r_safe, overlap_ratio,
    min_same_dist, min_cross_dist         # 新增：同向间距与异向间距
):
    """
    生成一条连续、大小均匀的圆柱墙。带有同向排斥逻辑，防止墙体粘连。
    返回 True 表示生成成功，False 表示被排斥放弃。
    """
    step_size = r_crash * 2 * overlap_ratio
    
    # 计算这面墙的起始和结束坐标
    if direction == 'H':
        dx, dy = (length - 1) * step_size, 0
    else:
        dx, dy = 0, (length - 1) * step_size
        
    x_start, y_start = cx - dx/2, cy - dy/2
    r_risk = r_crash + r_risk_offset

    # 1. 预先计算这堵墙所有圆柱的候选坐标
    candidate_points = []
    for i in range(length):
        x = x_start + (i * step_size if direction == 'H' else 0)
        y = y_start + (i * step_size if direction == 'V' else 0)
        candidate_points.append(np.array([x, y]))

    # 2. 合法性与排斥检查（只要有一个圆柱不合法，整堵墙都不要）
    for p in candidate_points:
        # a. 边界保护
        if not (r_risk <= p[0] <= Lx - r_risk and r_risk <= p[1] <= Ly - r_risk):
            return False

        # b. 安全通道检查
        if is_point_near_polyline(p, safe_waypoints, r_safe):
            return False

        # c. 【核心】同方向排斥检查 (保持较远距离)
        target_same = existing_H if direction == 'H' else existing_V
        if len(target_same) > 0:
            arr_same = np.array(target_same)
            dists = np.hypot(arr_same[:, 0] - p[0], arr_same[:, 1] - p[1])
            if np.any(dists < (r_crash * 2 + min_same_dist)):
                return False

        # d. 【核心】异方向排斥检查 (允许交叉或靠近)
        target_cross = existing_V if direction == 'H' else existing_H
        if len(target_cross) > 0:
            arr_cross = np.array(target_cross)
            dists = np.hypot(arr_cross[:, 0] - p[0], arr_cross[:, 1] - p[1])
            if np.any(dists < (r_crash * 2 + min_cross_dist)):
                return False

    # 3. 如果所有检查都通过，正式加入地图
    curr_zmax = np.random.uniform(zmax_range[0], min(zmax_range[1], Lz))
    for p in candidate_points:
        obstacle_list.append((p[0], p[1], 0.0, curr_zmax, r_crash, r_risk))
        # 记录到对应的坐标库中
        if direction == 'H':
            existing_H.append([p[0], p[1]])
        else:
            existing_V.append([p[0], p[1]])

    return True

# ==========================================
# 主生成函数：带有方向区分互斥的直角簇迷宫
# ==========================================
def env_generator_orthogonal_cluster_maze(
    map_dim=(1500, 1500, 240),
    r_crash_base=40,             # 固定的圆柱体半径
    r_risk_offset=15,            # 风险圈外扩大小
    zmax_range=(120, 240),
    density=0.15,                # 【新增】：墙体密度，推荐 0.1~0.4 之间调节地图难度
    chain_length_range=(3, 7),   # 每堵墙由几个圆柱组成
    overlap_ratio=0.85,          # 重叠系数
    safe_waypoints=None,         # 【修改】：如果为 None，则完全随机，不留保护通道
    r_safe_passage=160,          # 通道宽度
    min_same_dir_dist=120,       # 同方向墙壁的最小间距
    min_cross_dir_dist=10,       # 异方向墙壁的最小间距
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    Lx, Ly, Lz = map_dim
    obstacle_list = []
    
    existing_H_centers = []
    existing_V_centers = []
    
    # ==========================================
    # 1. 处理折线通道：为 None 则传入空列表，表示完全随机
    # ==========================================
    if safe_waypoints is None:
        waypoints_arr = []
    else:
        waypoints_arr = [np.array(wp, dtype=np.float32) for wp in safe_waypoints]
    
    # ==========================================
    # 2. 根据密度 (density) 动态计算需要生成的墙体数量
    # ==========================================
    # 基准容量算法：将地图按 100x100 的网格划分。
    # 对于 1500x1500 的地图，base_capacity = 225
    # 当 density = 0.2 时，大约生成 45 堵墙。这样换不同尺寸地图时难度会保持一致。
    base_capacity = (Lx / 100.0) * (Ly / 100.0)
    target_walls = int(density * base_capacity)
    
    margin = r_crash_base * 2
    
    # 动态调整最大尝试次数：密度越大，后期越难放置，需要更多尝试次数防止死循环
    max_attempts = max(2000, target_walls * 30) 
    walls_placed = 0

    for _ in range(max_attempts):
        if walls_placed >= target_walls:
            break  # 达到目标墙数就停止
            
        cx = np.random.uniform(margin, Lx - margin)
        cy = np.random.uniform(margin, Ly - margin)
        
        length = random.randint(*chain_length_range)
        direction = 'H' if random.random() < 0.5 else 'V'
        
        success = add_uniform_cylinder_wall_safe(
            obstacle_list, existing_H_centers, existing_V_centers,
            Lx, Ly, Lz, cx, cy, direction, length, 
            zmax_range, r_crash_base, r_risk_offset, 
            waypoints_arr, r_safe_passage, overlap_ratio,
            min_same_dir_dist, min_cross_dir_dist
        )
        
        if success:
            walls_placed += 1

    return {
        "map_dim": map_dim,
        "obstacles": obstacle_list,
        "num_obstacles": len(obstacle_list),
        "seed": seed,
        "density_used": density,          # 记录实际使用的密度
        "target_walls": target_walls,     # 记录目标生成的墙体数
        "walls_generated": walls_placed   # 记录实际成功生成的墙体数
    }
# -------测试-------
if __name__ == "__main__":

    env_map = env_generator(
            rho=random.uniform(0.6, 0.85),   # 数据更丰富不容易出现过拟合
            map_dim=(1500, 1500, 240),
            r_crash_range=(30, 50),
            r_risk_range=(3, 7),
            zmax_range=(30, 240),
            max_iter=5000,
            seed=42
        )

    env_map = env_generator_maze(
        grid_size=(4, 4),           # 4x4的网格，网格越多通道越窄越复杂
        map_dim=(1500, 1500, 240),
        r_crash_range=(40, 60),     # 为了给通道留出足够空间，半径相较于你原来设定的(80,125)稍微缩小了一些
        r_risk_range=(10, 20),
        zmax_range=(240, 240),
        seed=1
    )

#     env_map = env_generator_cluster(
#     map_dim=(1500, 1500, 240),   # (Lx, Ly, Lz)
#     num_clusters=20,             # 建议 10~15 之间，保证有足够空间
#     chain_length_range=(1, 4),   # 每个簇的圆柱体数量
#     r_center_range=(100, 150),    # 接近地图中心的圆柱体半径范围
#     r_edge_range=(20, 50),       # 接近地图边缘的圆柱体半径范围
#     r_risk_range=(10, 20),       # 风险半径偏移量
#     zmax_range=(240, 240),
#     min_center_dist=200,         # 【核心参数】任意两个簇中心点的最小绝对距离！
#     seed=None,
# )
    
    # env_map = env_generator_clutter(
    #         rho=0.7,   # 数据更丰富不容易出现过拟合
    #         map_dim=(1500, 1500, 240),
    #         r_crash_range=(30, 80),
    #         r_risk_range=(3, 7),
    #         zmax_range=(30, 240),
    #         max_iter=5000,
    #         seed=42
    #     )
    # FIXED_S = [0, 250, 120]
    # FIXED_G = [1500, 1250, 120]
    # my_custom_waypoints = [
    #         (FIXED_S[0], FIXED_S[1]),  # 第 1 个点：起点 (0, 250)
    #         (500, 750),                # 第 2 个点：左侧转折点
    #         (1000, 750),               # 第 3 个点：右侧转折点
    #         (FIXED_G[0], FIXED_G[1])   # 第 4 个点：终点 (1500, 1250)
    #     ]
    env_map = env_generator_orthogonal_cluster_maze(
            map_dim=(1500, 1500, 240),
            r_crash_base=40,             # 圆柱半径，统一为40
            r_risk_offset=10,            # 风险圈外扩大小
            zmax_range=(240, 240),
            density=0.12,                # 墙体密度，推荐 0.1~0.4 之间调节地图难度
            chain_length_range=(3, 10),   # 墙的长度，比如连续3到7个圆柱
            overlap_ratio=0.7,          # 让圆柱体紧密咬合
            safe_waypoints=None, # 传入Z型骨架
            r_safe_passage=80,          # 挖空的通道宽度
            min_same_dir_dist=60,       # 【关键参数】：同方向墙壁（横对横，竖对竖）的最小间距
            min_cross_dir_dist=-20,       # 【关键参数】：异方向墙壁（横对竖）的最小间距
            seed=None
        )


    plot_map(env_map)
