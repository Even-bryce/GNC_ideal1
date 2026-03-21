import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from res_show_for_data import plot_map

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


# -------测试-------
if __name__ == "__main__":

    env_map = env_generator(
        rho=0.6,
        map_dim=(1500, 1500, 240),
        r_crash_range=(30, 50),
        r_risk_range=(3, 7),
        zmax_range=(30, 240),
        max_iter=5000,
        seed=40
    )

    plot_map(env_map)
