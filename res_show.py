import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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


def plot_tree_and_path(env_map, node_list, path=None):
    """
    绘制最终结果树和路径 (包含圆柱体顶底盖)
    """
    obstacles = env_map["obstacles"]
    map_size = env_map["size"]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # ----------------------------------------
    # 1. 绘制障碍物 (侧面 + 顶底盖)
    # ----------------------------------------
    print("正在绘制环境 (含顶底盖)...")
    
    # --- A. 侧面网格 (Theta vs Z) ---
    theta = np.linspace(0, 2 * np.pi, 30) # 角度分辨率
    z_norm = np.linspace(0, 1, 2)         # 高度标准化 [0, 1]
    theta_grid, z_grid_norm = np.meshgrid(theta, z_norm)
    
    # --- B. 盖子网格 (Radius vs Theta) ---
    # 半径从 0 到 1，用于缩放
    r_norm = np.linspace(0, 1, 2)
    theta_grid_cap, r_grid_norm = np.meshgrid(theta, r_norm)

    for obs in obstacles:
        xc, yc, zmin, zmax, r_crash, r_risk = obs
        
        # =============================
        # Part 1: 绘制 Crash 区域 (深色, 不透明)
        # =============================
        # 1.1 侧面
        z_grid = z_grid_norm * (zmax - zmin) + zmin
        x_crash = xc + r_crash * np.cos(theta_grid)
        y_crash = yc + r_crash * np.sin(theta_grid)
        ax.plot_surface(x_crash, y_crash, z_grid, color='#555555', alpha=0.8, linewidth=0)
        
        # 1.2 顶面 (Lid) & 底面 (Bottom)
        # 计算盖子的 X, Y 坐标
        x_cap = xc + (r_grid_norm * r_crash) * np.cos(theta_grid_cap)
        y_cap = yc + (r_grid_norm * r_crash) * np.sin(theta_grid_cap)
        
        # 顶盖 Z=zmax
        z_cap_top = np.full_like(x_cap, zmax)
        ax.plot_surface(x_cap, y_cap, z_cap_top, color='#555555', alpha=0.8, linewidth=0)
        
        # 底盖 Z=zmin
        z_cap_bottom = np.full_like(x_cap, zmin)
        ax.plot_surface(x_cap, y_cap, z_cap_bottom, color='#555555', alpha=0.8, linewidth=0)

        # =============================
        # Part 2: 绘制 Risk 区域 (浅色, 透明)
        # =============================
        # 2.1 侧面
        x_risk = xc + r_risk * np.cos(theta_grid)
        y_risk = yc + r_risk * np.sin(theta_grid)
        ax.plot_surface(x_risk, y_risk, z_grid, color='#CCCCCC', alpha=0.25, linewidth=0)
        
        # 2.2 顶面 & 底面
        x_cap_risk = xc + (r_grid_norm * r_risk) * np.cos(theta_grid_cap)
        y_cap_risk = yc + (r_grid_norm * r_risk) * np.sin(theta_grid_cap)
        
        # 顶盖
        ax.plot_surface(x_cap_risk, y_cap_risk, z_cap_top, color='#CCCCCC', alpha=0.25, linewidth=0)
        # 底盖
        ax.plot_surface(x_cap_risk, y_cap_risk, z_cap_bottom, color='#CCCCCC', alpha=0.25, linewidth=0)

    # ----------------------------------------
    # 2. 绘制树和路径 (保持不变)
    # ----------------------------------------
    print(f"正在绘制路径 (节点数: {len(node_list)})...")
    
    # 绘制树枝
    for node in node_list:
        if node.parent:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.z, node.parent.z], 
                    color='lime', linewidth=1, alpha=0.5)

    if path is not None:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=3, label='Final Path')
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], c='blue', s=100, marker='^', label='Start')
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='orange', s=100, marker='*', label='Goal')

    # 设置轴和视角
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_zlim(0, env_map["z_size"])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("RRT* 3D Planning (Solid Cylinders)")
    ax.legend()
    
    # 调整视角以获得更好的立体感
    ax.view_init(elev=25, azim=-45)
    plt.show()