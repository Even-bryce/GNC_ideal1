import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from env_generator import env_generator

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
    map_size = env_map["size"]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 障碍物
    for obs in obstacles:
        xc, yc, zmin, zmax, r_crash, r_risk = obs
        # 碰撞区
        add_cylinder(ax, xc, yc, zmin, zmax, r_crash,
                     color='orange', alpha=0.8, wireframe=True, resolution=20)
        
    # # RRT树
    # if node_list is not None and len(node_list) > 1:
    #     segments = []
    #     for node in node_list:
    #         if node.parent:
    #             p_start = [node.x, node.y, node.z]
    #             p_end   = [node.parent.x, node.parent.y, node.parent.z]
    #             segments.append([p_start, p_end])
    #     if segments:
    #         segments = np.array(segments, dtype=float)
    #         # 使用 Line3DCollection 批量添加，效率高
    #         tree_collection = Line3DCollection(segments, colors='lime', linewidth=0.5, alpha=0.5)
    #         ax.add_collection3d(tree_collection)

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
    Lx, Ly = env_map["size"]
    Lz = env_map["z_size"]
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

    # ==========================
    # (1) ---- 3D 图（高效版） ----
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
    