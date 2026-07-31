import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from env_generator_for_data import env_generator, env_generator_maze, env_generator_cluster

def add_cylinder(ax, xc, yc, zmin, zmax, radius, color, alpha=1.0, wireframe=False, resolution=20):
    """
    向 3D 坐标轴添加圆柱体（使用多边形面片，高效）
    （复用之前的优化实现）
    """
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=True)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    bottom_pts = np.array([x, y, np.full_like(x, zmin)]).T
    top_pts = np.array([x, y, np.full_like(x, zmax)]).T

    faces = []
    for i in range(resolution - 1):
        quad = np.array([bottom_pts[i], bottom_pts[i+1], top_pts[i+1], top_pts[i]])
        faces.append(quad)
    quad_last = np.array([bottom_pts[-1], bottom_pts[0], top_pts[0], top_pts[-1]])
    faces.append(quad_last)
    top_face = np.vstack([top_pts, top_pts[0]])
    faces.append(top_face)
    bottom_face = np.vstack([bottom_pts, bottom_pts[0]])
    faces.append(bottom_face)

    if wireframe:
        # 线框模式：侧面竖线 + 上下圆环
        for i in range(resolution):
            ax.plot([x[i], x[i]], [y[i], y[i]], [zmin, zmax], color=color, linewidth=0.5, alpha=alpha)
        ax.plot(x, y, np.full_like(x, zmax), color=color, linewidth=0.5, alpha=alpha)
        ax.plot(x, y, np.full_like(x, zmin), color=color, linewidth=0.5, alpha=alpha)
    else:
        collection = Poly3DCollection(faces, facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_collection3d(collection)


def plot_tree_and_path(env_map, node_list=None, path=None, waypoint_list=None):
    """
    绘制障碍物、RRT 树、最终路径、航路点
    """
    obstacles = env_map["obstacles"]
    map_size = env_map["map_dim"][0]  # 假设为正方体，Z范围另行获取

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 障碍物
    for obs in obstacles:
        xc, yc, zmin, zmax, r_crash, r_risk = obs
        # 碰撞区
        add_cylinder(ax, xc, yc, zmin, zmax, r_crash,
                     color='orange', alpha=0.8, wireframe=True, resolution=20)
        
    # # RRT树
    # all_segments = []
    # for seg_idx, tree_nodes in enumerate(node_list):
    #     if not tree_nodes or len(tree_nodes) < 2:
    #         continue
    #     seg_segments = []
    #     for node in tree_nodes:
    #         if node.parent is not None:
    #             p_start = [node.x, node.y, node.z]
    #             p_end   = [node.parent.x, node.parent.y, node.parent.z]
    #             seg_segments.append([p_start, p_end])
    #     if seg_segments:
    #         all_segments.extend(seg_segments)   # 或者按航段分别添加，便于不同颜色
    
    # if not all_segments:
    #     return
    
    # segments_array = np.array(all_segments, dtype=float)
    # # 可依据需要为每个航段单独设置颜色：colors=color_list[seg_idx]
    # tree_collection = Line3DCollection(segments_array, colors='lime', linewidth=0.5, alpha=0.5)
    # ax.add_collection3d(tree_collection)

    # 最终路径
    if path is not None:
        path_arr = np.array(path, dtype=float)
        if len(path_arr) > 1:
            # 路径用红色粗线
            ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                    color='red', linewidth=3, label='Final Path')
            # 起点和终点用球体
            ax.scatter(*path_arr[0], color='blue', s=100, label='Start', depthshade=True)
            ax.scatter(*path_arr[-1], color='orange', s=100, label='Goal', depthshade=True)
        
    # 航路点
    if waypoint_list is not None and len(waypoint_list) > 0:
        wp = np.array(waypoint_list, dtype=float)
        # 航路点球体
        ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2],
                   color='purple', s=80, label='Waypoints', depthshade=True)

    # ---------- 4. 地图边界和装饰 ----------
    Lx, Ly, Lz = env_map["map_dim"]
    # 绘制底面边界框
    bx = [0, Lx, Lx, 0, 0]
    by = [0, 0, Ly, Ly, 0]
    bz = [0, 0, 0, 0, 0]
    ax.plot(bx, by, bz, 'k-', linewidth=2)
    # 垂直棱线
    for x in [0, Lx]:
        for y in [0, Ly]:
            ax.plot([x, x], [y, y], [0, Lz], 'k-', linewidth=1, alpha=0.5)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D RRT Tree and Final Path (Matplotlib)")
    ax.view_init(elev=90, azim=0)

    # 添加图例（自定义代理）
    legend_elements = [
        Line2D([0], [0], color='lime', linewidth=2, label='RRT Tree'),
        Line2D([0], [0], color='red', linewidth=3, label='Final Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Goal'),
        Line2D([0], [0], color='gray', linewidth=2, label='Obstacle (Crash)'),
        Line2D([0], [0], color='orange', linewidth=1, linestyle='--', label='Risk Zone'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_map(env_map):
    """
    绘制地图：
    ① 3D 图 - 显示 r_crash 和 r_risk 的柱体
    ② XY 俯视图 - 显示两个半径的圆形投影
    """
    obstacles = env_map["obstacles"]
    map_size = env_map["size"]

    fig = plt.figure(figsize=(14, 6))

    # ==========================
    # (1) ---- 3D 图 ----
    # ==========================
    ax1 = fig.add_subplot(121, projection='3d')

    for obs in obstacles:
        xc, yc, zmin, zmax, r_crash, r_risk = obs

        theta = np.linspace(0, 2 * np.pi, 40)
        z = np.linspace(zmin, zmax, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)

        # 风险柱体（外层）
        x_risk = xc + r_risk * np.cos(theta_grid)
        y_risk = yc + r_risk * np.sin(theta_grid)
        ax1.plot_surface(x_risk, y_risk, z_grid,
                        color='gray', alpha=0.2, linewidth=0)

        # 碰撞柱体（内层）
        x_crash = xc + r_crash * np.cos(theta_grid)
        y_crash = yc + r_crash * np.sin(theta_grid)
        ax1.plot_surface(x_crash, y_crash, z_grid,
                        color='gray', alpha=0.7, linewidth=0)

    # 地图边界
    bx = [0, map_size, map_size, 0, 0]
    by = [0, 0, map_size, map_size, 0]
    bz = [0, 0, 0, 0, 0]
    ax1.plot(bx, by, bz, 'k-', linewidth=2)

    ax1.set_xlim(0, map_size)
    ax1.set_ylim(0, map_size)
    max_z = max([obs[3] for obs in obstacles]) if obstacles else 200
    ax1.set_zlim(0, max_z)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Cylindrical Obstacle Map")

    # ==========================
    # (2) ---- XY 俯视图 ----
    # ==========================
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')

    for obs in obstacles:
        xc, yc, _, _, r_crash, r_risk = obs

        # 风险区圆（外圈）
        circle_risk = plt.Circle((xc, yc), r_risk,
                                color='gray', alpha=0.2, linewidth=1)
        ax2.add_patch(circle_risk)

        # 碰撞区圆（内圈）
        circle_crash = plt.Circle((xc, yc), r_crash,
                                color='gray', alpha=0.7, linewidth=1)
        ax2.add_patch(circle_crash)

    # 地图边界
    ax2.plot(bx, by, 'k-', linewidth=2)
    ax2.set_xlim(0, map_size)
    ax2.set_ylim(0, map_size)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top-Down View (XY Projection)")

    plt.tight_layout()
    plt.show()

def plot_map_and_waypoint(env_map, waypoints):
    """
    优化后的地图绘制函数：
    ① 3D 图 - 碰撞区半透明实心圆柱体，风险区线框圆柱体
    ② XY 俯视图 - 圆形投影（保持原有效果）
    """
    obstacles = env_map["obstacles"]
    map_size = env_map["map_dim"][0]

    fig = plt.figure(figsize=(14, 6))

    # # ==========================
    # # (1) ---- 3D 图（高效版） ----
    # # ==========================
    # ax1 = fig.add_subplot(121, projection='3d')

    # for obs in obstacles:
    #     xc, yc, zmin, zmax, r_crash, r_risk = obs

    #     theta = np.linspace(0, 2 * np.pi, 40)
    #     z = np.linspace(zmin, zmax, 20)
    #     theta_grid, z_grid = np.meshgrid(theta, z)

    #     # 风险柱体（外层）
    #     x_risk = xc + r_risk * np.cos(theta_grid)
    #     y_risk = yc + r_risk * np.sin(theta_grid)
    #     ax1.plot_surface(x_risk, y_risk, z_grid,
    #                     color='gray', alpha=0.2, linewidth=0)

    #     # 碰撞柱体（内层）
    #     x_crash = xc + r_crash * np.cos(theta_grid)
    #     y_crash = yc + r_crash * np.sin(theta_grid)
    #     ax1.plot_surface(x_crash, y_crash, z_grid,
    #                     color='gray', alpha=0.7, linewidth=0)

    # # 地图边界
    bx = [0, map_size, map_size, 0, 0]
    by = [0, 0, map_size, map_size, 0]
    bz = [0, 0, 0, 0, 0]
    # ax1.plot(bx, by, bz, 'k-', linewidth=2)

    # ax1.set_xlim(0, map_size)
    # ax1.set_ylim(0, map_size)
    # max_z = max([obs[3] for obs in obstacles]) if obstacles else 200
    # ax1.set_zlim(0, max_z)

    # ax1.set_xlabel("X")
    # ax1.set_ylabel("Y")
    # ax1.set_zlabel("Z")
    # ax1.set_title("3D Cylindrical Obstacle Map")

    # ==========================
    # XY 俯视图
    # ==========================
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')

    for obs in obstacles:
        xc, yc, _, _, r_crash, r_risk = obs

        # 风险区圆（外圈，浅色）
        circle_risk = Circle((xc, yc), r_risk,
                             color='gray', alpha=0.2, linewidth=1, fill=True)
        ax2.add_patch(circle_risk)

        # 碰撞区圆（内圈，深色）
        circle_crash = Circle((xc, yc), r_crash,
                              color='gray', alpha=0.7, linewidth=1, fill=True)
        ax2.add_patch(circle_crash)
        
    if waypoints is not None and len(waypoints) >= 2:
        xs = [pt[0] for pt in waypoints]
        ys = [pt[1] for pt in waypoints]
        # 绘制航路点
        ax2.scatter(xs, ys, c='red', marker='o', s=60, zorder=5)
        ax2.legend()

    # 地图边界
    ax2.plot(bx, by, 'k-', linewidth=2)
    ax2.set_xlim(0, map_size)
    ax2.set_ylim(0, map_size)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top-Down View (XY Projection)")

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    # 生成地图

    # env_map = env_generator(
    #     rho=0.4,
    #     map_dim=(1500, 1500, 240),
    #     r_crash_range=(30, 50),
    #     r_risk_range=(3, 7),
    #     zmax_range=(30, 240),
    #     max_iter=10000,
    #     seed=1
    # )
    # waypoints = [[0, 0, 0], [332, 467, 14], [616, 802, 24], [1027, 1028, 77], [1237, 1277, 67], [1500, 1500, 100]]
    
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
    # waypoints = [[0, 0, 0], [360, 806, 54], [832, 1415, 52], [1109, 1491, 85], [1500, 1500, 100]]
    
    
    
    env_map = env_generator_maze(
        grid_size=(4, 4),           # 4x4的网格，网格越多通道越窄越复杂
        map_dim=(1500, 1500, 240),
        r_crash_range=(40, 60),     # 为了给通道留出足够空间，半径相较于你原来设定的(80,125)稍微缩小了一些
        r_risk_range=(10, 20),
        zmax_range=(240, 240),
        seed=42
    )
    waypoints = [[0, 200, 0], [551, 362, 33], [1136, 695, 36], [1072, 1163, 73], [1500, 1300, 100]]
    plot_map_and_waypoint(env_map, waypoints)