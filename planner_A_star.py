import numpy as np
import heapq
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle
import warnings

# ===================== 环境生成器（已提供，稍作修改以支持三维） =====================
def env_generator(
    rho=0.8,
    map_dim=(1500, 1500, 300),
    r_crash_range=(30, 50),
    r_risk_range=(3, 7),
    zmax_range=(30, 240),
    max_iter=5000,
    seed=None,
):
    """
    生成随机圆柱障碍物地图
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
            r_crash = np.random.uniform(*r_crash_range)
            r_risk = r_crash + np.random.uniform(*r_risk_range)
            x = np.random.uniform(r_risk, Lx - r_risk)
            y = np.random.uniform(r_risk, Ly - r_risk)
            zmin = 0.0
            zmax = np.random.uniform(zmax_range[0], min(zmax_range[1], Lz))
            obstacle_list.append((x, y, zmin, zmax, r_crash, r_risk))
            break
        else:
            print("Warning: reached max_iter, some obstacles not placed.")
            break

    map_dict = {
        "map_dim": map_dim,
        "rho": rho,
        "area": map_area,
        "obstacles": obstacle_list,
        "num_obstacles": len(obstacle_list),
        "seed": seed,
    }
    return map_dict


# ===================== 3D A* 路径规划 =====================
def a_star_planning_3d(map_dict, step_size, start, goal):
    """
    三维 A* 路径规划（26 邻域）

    参数：
        map_dict : 环境字典，包含 'map_dim' 和 'obstacles'
        step_size : 栅格边长（三个维度统一）
        start     : 起点坐标 (x, y, z)
        goal      : 终点坐标 (x, y, z)

    返回：
        result 字典，包含：
            'path'          : 路径点列表 [(x,y,z), ...]
            'path_length'   : 路径总长度（三维欧氏距离累加）
            'time'          : 搜索耗时（秒）
            'num_steps'     : 路径段数
            'found'         : 是否找到路径
            'obstacle_grid' : 三维布尔型栅格数组 (nx, ny, nz)（用于调试/可视化）
            'step_size'     : 步长
            'start_index'   : 起点栅格索引
            'goal_index'    : 终点栅格索引
    """
    obstacles = map_dict['obstacles']
    Lx, Ly, Lz = map_dict['map_dim']

    # ---------- 1. 生成三维栅格并标记障碍 ----------
    nx = int(np.ceil(Lx / step_size))
    ny = int(np.ceil(Ly / step_size))
    nz = int(np.ceil(Lz / step_size))
    obstacle_grid = np.zeros((nx, ny, nz), dtype=bool)

    # 预计算每个体素中心坐标
    xs = (np.arange(nx) + 0.5) * step_size
    ys = (np.arange(ny) + 0.5) * step_size
    zs = (np.arange(nz) + 0.5) * step_size

    # 对于每个障碍物，快速标记受影响的体素（避免三层循环，使用向量化）
    # 但为了清晰，这里采用三层循环（对于中等规模足够）
    for i in range(nx):
        cx = xs[i]
        for j in range(ny):
            cy = ys[j]
            for k in range(nz):
                cz = zs[k]
                blocked = False
                for ox, oy, zmin, zmax, r_crash, _ in obstacles:
                    # 检查高度是否在圆柱范围内
                    if cz < zmin or cz > zmax:
                        continue
                    # 检查水平距离
                    if (cx - ox)**2 + (cy - oy)**2 < r_crash**2:
                        blocked = True
                        break
                obstacle_grid[i, j, k] = blocked

    # ---------- 2. 起点/终点映射到栅格索引 ----------
    def coord_to_index(coord, axis_max):
        idx = int(np.floor(coord / step_size))
        return max(0, min(idx, axis_max - 1))

    si, sj, sk = coord_to_index(start[0], nx), coord_to_index(start[1], ny), coord_to_index(start[2], nz)
    gi, gj, gk = coord_to_index(goal[0], nx), coord_to_index(goal[1], ny), coord_to_index(goal[2], nz)

    # 检查起点终点是否被占据
    if obstacle_grid[si, sj, sk]:
        raise ValueError(f"起点 {start} 所在栅格被障碍物占据")
    if obstacle_grid[gi, gj, gk]:
        raise ValueError(f"终点 {goal} 所在栅格被障碍物占据")

    # ---------- 3. 26 邻域定义 ----------
    neighbors = [(di, dj, dk) for di in [-1, 0, 1] for dj in [-1, 0, 1] for dk in [-1, 0, 1]
                 if not (di == 0 and dj == 0 and dk == 0)]

    # 预计算每个邻居的移动代价（对于统一步长）
    move_costs = {}
    for di, dj, dk in neighbors:
        # 欧氏距离乘以步长
        cost = step_size * np.sqrt(di**2 + dj**2 + dk**2)
        move_costs[(di, dj, dk)] = cost

    # ---------- 4. 启发函数（三维欧氏距离） ----------
    def heuristic(i, j, k):
        return np.sqrt((i - gi)**2 + (j - gj)**2 + (k - gk)**2) * step_size

    # ---------- 5. A* 搜索 ----------
    open_set = []
    heapq.heappush(open_set, (0.0, si, sj, sk))
    g_score = np.full((nx, ny, nz), np.inf)
    g_score[si, sj, sk] = 0.0
    f_score = np.full((nx, ny, nz), np.inf)
    f_score[si, sj, sk] = heuristic(si, sj, sk)
    came_from = {}

    start_time = time.time()
    found = False
    nodes_expanded = 0

    while open_set:
        _, i, j, k = heapq.heappop(open_set)
        nodes_expanded += 1

        if (i, j, k) == (gi, gj, gk):
            found = True
            break

        for di, dj, dk in neighbors:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                continue
            if obstacle_grid[ni, nj, nk]:
                continue

            move_cost = move_costs[(di, dj, dk)]
            tentative_g = g_score[i, j, k] + move_cost
            if tentative_g < g_score[ni, nj, nk]:
                came_from[(ni, nj, nk)] = (i, j, k)
                g_score[ni, nj, nk] = tentative_g
                f_score[ni, nj, nk] = tentative_g + heuristic(ni, nj, nk)
                heapq.heappush(open_set, (f_score[ni, nj, nk], ni, nj, nk))

    elapsed = time.time() - start_time

    # ---------- 6. 重构路径 ----------
    if not found:
        return {
            'path': [],
            'path_length': np.inf,
            'time': elapsed,
            'num_steps': 0,
            'found': False,
            'obstacle_grid': obstacle_grid,
            'step_size': step_size,
            'nodes_expanded': nodes_expanded,
        }

    # 回溯
    path_idx = []
    cur = (gi, gj, gk)
    while cur in came_from:
        path_idx.append(cur)
        cur = came_from[cur]
    path_idx.append((si, sj, sk))
    path_idx.reverse()

    # 转换为实际坐标（体素中心）
    path_coords = []
    for i, j, k in path_idx:
        x = (i + 0.5) * step_size
        y = (j + 0.5) * step_size
        z = (k + 0.5) * step_size
        path_coords.append((x, y, z))

    # 计算路径长度（三维欧氏距离累加）
    path_length = 0.0
    for idx in range(len(path_coords) - 1):
        x1, y1, z1 = path_coords[idx]
        x2, y2, z2 = path_coords[idx + 1]
        path_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    return {
        'path': path_coords,
        'path_length': path_length,
        'time': elapsed,
        'num_steps': len(path_coords) - 1,
        'found': True,
        'obstacle_grid': obstacle_grid,
        'step_size': step_size,
        'start_index': (si, sj, sk),
        'goal_index': (gi, gj, gk),
        'nodes_expanded': nodes_expanded,
    }


# ===================== 三维可视化辅助函数 =====================
def draw_cylinder(ax, x, y, zmin, zmax, radius, color='red', alpha=0.3, resolution=20):
    """
    在3D坐标轴上绘制一个圆柱体（透明）
    """
    # 生成圆柱体表面数据
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(zmin, zmax, 2)
    U, V = np.meshgrid(u, v)
    X = x + radius * np.cos(U)
    Y = y + radius * np.sin(U)
    Z = V

    # 绘制侧面
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, edgecolor='none')

    # 绘制顶底圆（可选）
    if zmin < zmax:
        # 底部圆
        theta = np.linspace(0, 2*np.pi, resolution)
        x_circle = x + radius * np.cos(theta)
        y_circle = y + radius * np.sin(theta)
        ax.plot(x_circle, y_circle, zmin * np.ones_like(theta), color=color, alpha=alpha)
        ax.plot(x_circle, y_circle, zmax * np.ones_like(theta), color=color, alpha=alpha)


def visualize_path_3d(map_dict, result, start, goal, show_grid=False):
    """
    三维可视化：显示圆柱体障碍物、起点、终点和规划路径
    """
    obstacles = map_dict['obstacles']
    Lx, Ly, Lz = map_dict['map_dim']

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制每个圆柱体障碍物
    for obs in obstacles:
        ox, oy, zmin, zmax, r_crash, _ = obs
        draw_cylinder(ax, ox, oy, zmin, zmax, r_crash, color='red', alpha=0.2)

    # 绘制起点和终点
    ax.scatter(*start, color='green', s=100, label='Start', edgecolors='black', linewidth=1.5)
    ax.scatter(*goal, color='blue', s=100, label='Goal', edgecolors='black', linewidth=1.5)

    # 绘制路径
    if result['found'] and result['path']:
        xs = [p[0] for p in result['path']]
        ys = [p[1] for p in result['path']]
        zs = [p[2] for p in result['path']]
        ax.plot(xs, ys, zs, color='cyan', linewidth=3, label='3D A* Path')
        # 绘制路径点（可选）
        ax.scatter(xs[1:-1], ys[1:-1], zs[1:-1], color='cyan', s=20, alpha=0.6)

    # 设置坐标轴
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D A* Path Planning in Cylinder Obstacle Environment')
    ax.legend()
    ax.view_init(elev=30, azim=45)  # 调整视角

    plt.tight_layout()
    plt.show()

def load_test_case(filename):
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)
    
    env_map = loaded_data["env_map"]
    final_waypoints = loaded_data["final_waypoints"]
    return env_map, final_waypoints    

if __name__ == "__main__":
    # map_dict, waypoints = load_test_case('test_case_cluster.pkl')
    map_dict, waypoints = load_test_case('test_case_clutter.pkl')
    # map_dict, waypoints = load_test_case('test_case_maze.pkl')

    # 2. 设置起点、终点
    start = waypoints[0]
    goal  = waypoints[-1]

    # 3. 执行3D A*规划（步长设为25）
    step = 15
    result = a_star_planning_3d(map_dict, step, start, goal)

    # 4. 输出指标
    if result['found']:
        print("========== 规划结果 ==========")
        print(f"路径规划耗时: {result['time']:.4f} 秒")
        print(f"路径长度: {result['path_length']:.2f} 单位")
        print(f"路径步数: {result['num_steps']}")
        print(f"扩展节点数: {result['nodes_expanded']}")
        print(f"起点栅格索引: {result['start_index']}")
        print(f"终点栅格索引: {result['goal_index']}")
        print("==============================")
    else:
        print("未找到可行路径！")
        print(f"搜索耗时: {result['time']:.4f} 秒")
        print(f"扩展节点数: {result['nodes_expanded']}")

    # 5. 三维可视化
    visualize_path_3d(map_dict, result, start, goal)