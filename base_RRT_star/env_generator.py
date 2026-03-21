import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from res_show import plot_map

def env_generator( 
    rho=0.8,
    map_size=1500,
    r_crash_range=(30, 50),      # 碰撞半径
    r_risk_range=(3, 7),         # 风险半径偏移量（基于 r_crash）
    zmax_range=(30, 240),
    z_size=300,
    max_iter=5000,
    seed=None,
):
    """
    生成随机圆柱障碍物地图（允许交叉）
    r_risk = r_crash + 随机偏移量
    """

    if seed is not None:
        np.random.seed(seed)

    obstacle_list = []
    map_area = map_size * map_size

    r_avg = (r_crash_range[0] + r_crash_range[1]) / 2
    obs_area_avg = np.pi * r_avg**2
    num_obs_est = int(rho * map_area / obs_area_avg)

    for _ in range(num_obs_est):
        for _ in range(max_iter):

            # 先生成碰撞半径（内圈）
            r_crash = np.random.uniform(r_crash_range[0], r_crash_range[1])

            # 生成风险半径（外圈 = 冲撞 + 偏移）
            r_risk = r_crash + np.random.uniform(r_risk_range[0], r_risk_range[1])

            # ⚠ 位置必须保证外圈圆柱体完全在地图内部
            x = np.random.uniform(r_risk, map_size - r_risk)
            y = np.random.uniform(r_risk, map_size - r_risk)

            zmin = 0
            zmax = np.random.uniform(zmax_range[0], zmax_range[1])

            obstacle_list.append((x, y, zmin, zmax, r_crash, r_risk))
            break

        else:
            print("Warning: reached max_iter, some obstacles not placed.")
            break

    map_dict = {
        "size": map_size,
        "z_size": z_size,
        "rho": rho,
        "area": map_area,
        "obstacles": obstacle_list,
        "num_obstacles": len(obstacle_list),
        "seed": seed,
    }

    return map_dict



if __name__ == "__main__":
    
    env_map = env_generator(
        rho=0.6, 
        map_size=1500,
        r_crash_range=(30, 50),
        r_risk_range=(3, 7),
        zmax_range=(30, 240),
        z_size=240,
        max_iter=5000,
        seed=40
    )
    plot_map(env_map)