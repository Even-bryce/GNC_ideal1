import math
import random
from env_generator_for_data import env_generator, env_generator_cluster, env_generator_maze

def is_point_in_risk(point, obs):
    """检查一个点是否在障碍物的风险圆柱内（含碰撞区）"""
    x, y, z = point
    xc, yc, zmin, zmax, r_crash, r_risk = obs
    if zmin <= z <= zmax:
        if math.hypot(x - xc, y - yc) <= r_crash:
            # print(point)
            return True
    return False

def all_waypoints_safe(waypoints, obstacles):
    """判断所有航路点是否均不落入任何障碍物风险区"""
    for pt in waypoints:
        for obs in obstacles:
            if is_point_in_risk(pt, obs):
                return False
    return True

def random_offset_waypoints(env_map, waypoints, num_groups, offset_radius, max_attempts=200):
    """
    生成多组无碰撞的随机偏移航路点（起点和终点不变，中间点在三维球体内随机偏移）。
    输出航路点坐标均为整数（四舍五入取整）。

    参数
    ----------
    env_seed : int
        地图生成的随机种子
    env_params : dict
        传递给 env_generator 的环境参数（不含 seed）
    waypoints : list of [x, y, z]
        原始航路点列表（必须本身与地图无碰撞，否则函数会抛出异常）
    num_groups : int
        需要生成的偏移航路点组数（每组是一个完整的航路点列表）
    offset_radius : float
        三维球体偏移的最大半径（米）
    max_attempts : int
        为单个中间点寻找安全偏移位置的最大尝试次数，超出则保留原始点并打印警告

    返回
    -------
    list of list
        共 num_groups 组航路点，每组为列表形式：[[x1,y1,z1], [x2,y2,z2], ...]，坐标均为整数
    """
    # 生成固定地图
    obstacles = env_map["obstacles"]

    # 检查原始航路点是否安全
    if not all_waypoints_safe(waypoints, obstacles):
        raise ValueError("原始航路点中存在与障碍物风险区碰撞的点，无法进行安全偏移。")

    def is_safe(pt):
        for obs in obstacles:
            if is_point_in_risk(pt, obs):
                return False
        return True

    results = []
    n_waypoints = len(waypoints)
    if n_waypoints < 2:
        # 如果没有中间点，直接返回 num_groups 份原始点（取整）
        return [[[round(c) for c in p] for p in waypoints] for _ in range(num_groups)]

    for _ in range(num_groups):
        new_wp = [list(waypoints[0])]          # 起点不变
        # 对中间点进行偏移
        for idx in range(1, n_waypoints - 1):
            orig_x, orig_y, orig_z = waypoints[idx]
            success = False
            for attempt in range(max_attempts):
                # 在三维球体内均匀随机偏移（体积均匀）
                # 方法：半径 = radius * 立方根(随机数)，方向均匀分布在球面上
                r = offset_radius * (random.random() ** (1.0/3.0))
                # 随机方向（标准正态分布归一化）
                theta = random.uniform(0, 2 * math.pi)      # 方位角
                phi = random.uniform(0, math.pi)            # 极角 (0~pi)
                dx = r * math.sin(phi) * math.cos(theta)
                dy = r * math.sin(phi) * math.sin(theta)
                dz = r * math.cos(phi)

                new_pt_float = [orig_x + dx, orig_y + dy, orig_z + dz]
                # 先检查浮点版本是否安全
                if is_safe(new_pt_float):
                    # 取整为整数
                    int_pt = [round(new_pt_float[0]), round(new_pt_float[1]), round(new_pt_float[2])]
                    # 再检查取整后的点是否安全（因取整可能移动到风险区内）
                    if is_safe(int_pt):
                        new_wp.append(int_pt)
                        success = True
                        break
            if not success:
                # 保留原始点（取整）
                print(f"警告：航路点索引 {idx} 在 {max_attempts} 次尝试后未找到安全整数偏移位置，保留原始整数点。")
                new_wp.append([round(orig_x), round(orig_y), round(orig_z)])
        new_wp.append(list(waypoints[-1]))      # 终点不变
        results.append(new_wp)
    return results

# ============ 使用示例 ============
if __name__ == "__main__":
    # 初始航路点（可为整数或浮点数）
    waypoints = [[0, 200, 0], [551, 362, 33], [1136, 695, 36], [1072, 1163, 73], [1500, 1300, 100]]

    # env_map = env_generator_cluster(
    #     map_dim=(1500, 1500, 240),   # (Lx, Ly, Lz)
    #     num_clusters=20,             # 建议 10~15 之间，保证有足够空间
    #     chain_length_range=(1, 4),   # 每个簇的圆柱体数量
    #     r_center_range=(100, 150),    # 接近地图中心的圆柱体半径范围
    #     r_edge_range=(20, 50),       # 接近地图边缘的圆柱体半径范围
    #     r_risk_range=(10, 20),       # 风险半径偏移量
    #     zmax_range=(240, 240),
    #     min_center_dist=200,         # 【核心参数】任意两个簇中心点的最小绝对距离！
    #     seed=7,
    # )  
    env_map = env_generator_maze(
        grid_size=(4, 4),           # 4x4的网格，网格越多通道越窄越复杂
        map_dim=(1500, 1500, 240),
        r_crash_range=(40, 60),     # 为了给通道留出足够空间，半径相较于你原来设定的(80,125)稍微缩小了一些
        r_risk_range=(10, 20),
        zmax_range=(240, 240),
        seed=42
    )

    offset_groups = random_offset_waypoints(
        env_map=env_map,
        waypoints=waypoints,
        num_groups=19,
        offset_radius=50,
        max_attempts=200
    )

    print(f"成功生成 {len(offset_groups)} 组整数坐标航路点：")
    for i, group in enumerate(offset_groups):
        print(group)
    