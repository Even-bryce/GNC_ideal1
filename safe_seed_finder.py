import math
import numpy as np
from env_generator_for_data import env_generator

def is_point_in_risk(point, obs):
    """检查一个点是否在障碍物的风险圆柱内（含碰撞区）"""
    x, y, z = point
    xc, yc, zmin, zmax, _, r_risk = obs       # 只需要风险半径
    if zmin <= z <= zmax:
        if math.hypot(x - xc, y - yc) <= r_risk:
            return True
    return False

def all_waypoints_safe(waypoints, obstacles):
    """判断所有航路点是否均不落入任何障碍物风险区"""
    for pt in waypoints:
        for obs in obstacles:
            if is_point_in_risk(pt, obs):
                return False
    return True

def find_safe_seeds(waypoints, n=50, start_seed=0, env_params=None, max_attempts=5000):
    """
    遍历种子，寻找前 n 个使航路点完全避开障碍物风险区的地图种子。

    参数
    ----------
    waypoints : list of [x, y, z]
    n : int
        需要找到的安全种子数量
    start_seed : int
        起始种子编号
    env_params : dict
        传递给 env_generator 的环境参数（不含 seed）
    max_attempts : int
        最多尝试的种子数，防止无限循环
    """
    if env_params is None:
        env_params = {
            "rho": 0.4,
            "map_dim": (1500, 1500, 240),
            "r_crash_range": (30, 50),
            "r_risk_range": (3, 7),
            "zmax_range": (30, 240),
            "max_iter": 10000
        }

    safe_seeds = []
    seed = start_seed

    while len(safe_seeds) < n and (seed - start_seed) < max_attempts:
        # 用当前种子生成地图
        env_map = env_generator(seed=seed, **env_params)
        if all_waypoints_safe(waypoints, env_map["obstacles"]):
            safe_seeds.append(seed)
        seed += 1

    if len(safe_seeds) < n:
        print(f"仅在 {max_attempts} 次尝试中找到 {len(safe_seeds)} 个安全种子。")
    return safe_seeds

# ============ 使用示例 ============
if __name__ == "__main__":
    # 航路点
    waypoints = [[0, 0, 0],
                 [900, 600, 0],
                 [1500, 1500, 100]]

    # 环境参数
    env_params = {
        "rho": 0.4,
        "map_dim": (1500, 1500, 240),
        "r_crash_range": (30, 50),
        "r_risk_range": (3, 7),
        "zmax_range": (30, 240),
        "max_iter": 10000
    }

    n_safe = 58
    safe_list = find_safe_seeds(waypoints, n=n_safe, env_params=env_params)
    print(f"找到 {len(safe_list)} 个安全种子：{safe_list}")