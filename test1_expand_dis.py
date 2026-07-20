import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process
import os
from env_generator_for_data import env_generator, env_generator_maze, env_generator_cluster
from planner_VRRT import VRRT
from planner_VRRT_star import VRRT_star
from planner_VRRT_star_APF import VRRT_star_APF
from planner_VRRT_star_Bi import VRRT_star_Bi
from planner_VRRT_star_Bi_Informed import VRRT_star_Bi_Informed

def calculate_path_smoothness(path):
    """
    计算由一系列坐标点组成的路径的平滑度。

    参数:
        path: list of list or array-like，每个元素为 [x, y, z] 坐标。
              路径按顺序连接，即 path[0] -> path[1] -> path[2] -> ... 形成连续线段。

    返回:
        float: 相邻线段之间角度偏差的平均值（单位：度）。
               如果路径节点少于3个，返回 0.0。
    """
    if len(path) < 3:
        return 0.0

    angles = []
    for i in range(len(path) - 2):
        p1 = path[i]
        p2 = path[i + 1]
        p3 = path[i + 2]

        # 向量 v1: p1 -> p2
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
        # 向量 v2: p2 -> p3
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # 跳过长度为零的线段（重复点）
        if norm1 == 0 or norm2 == 0:
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)

    if not angles:
        return 0.0

    return float(np.degrees(np.mean(angles)))

def evaluate_step_length(args):
    step_length, env_map, waypoints, r_agent_crash, r_agent_risk, n_tests, planner_type = args

    pid = current_process().pid
    np.random.seed((int(step_length * 1000) + pid * 997) % 2**31)

    obstacles = env_map["obstacles"]

    # 根据规划器类型实例化
    if planner_type == 'VRRT':
        rrt = VRRT(env_map=env_map, 
                   waypoints=waypoints,
                   R_crash=r_agent_crash, 
                   R_risk=r_agent_risk,
                   obstacle_list=obstacles, 
                   expand_dis=step_length, 
                   max_iter=10000)
    elif planner_type == 'VRRT_star':
        rrt = VRRT_star(env_map=env_map, 
                        waypoints=waypoints,
                        R_crash=r_agent_crash, 
                        R_risk=r_agent_risk,
                        obstacle_list=obstacles, 
                        expand_dis=step_length,
                        max_iter=2000, 
                        search_radius=100, 
                        search_until_max_iter=True)
    elif planner_type == 'VRRT_star_APF':
        rrt = VRRT_star_APF(env_map=env_map, 
                            waypoints=waypoints,
                            R_crash=r_agent_crash, 
                            R_risk=r_agent_risk,
                            obstacle_list=obstacles, 
                            expand_dis=step_length,
                            max_iter=2000, 
                            search_radius=100, 
                            search_until_max_iter=True)
    elif planner_type == 'VRRT_star_Bi':
        rrt = VRRT_star_Bi(env_map=env_map, 
                           waypoints=waypoints,
                           R_crash=r_agent_crash, 
                           R_risk=r_agent_risk,
                           obstacle_list=obstacles, 
                           expand_dis=step_length,
                           max_iter=2000, 
                           search_radius=100, 
                           search_until_max_iter=True)
    elif planner_type == 'VRRT_star_Bi_Informed':
        rrt = VRRT_star_Bi_Informed(env_map=env_map, 
                                    waypoints=waypoints,
                                    R_crash=r_agent_crash, 
                                    R_risk=r_agent_risk,
                                    obstacle_list=obstacles, 
                                    expand_dis=step_length,
                                    max_iter=2000, 
                                    search_radius=100, 
                                    search_until_max_iter=True)
    else:
        raise ValueError(f"未知的规划器类型: {planner_type}")

    # 收集每次测试的详细指标
    records = {
        'max_time_first': [],
        'len_first': [],
        'smooth_first': [],
        'sum_iter': [],
        'time_final': [],
        'len_final': [],
        'smooth_final': []
    }

    success_count = 0

    for test_id in range(n_tests):
        t_start = time.time()
        result = rrt.planning()
        t_end = time.time()

        if result[0] is None:   # 整体失败
            continue

        # 解包返回值
        (first_path_found, time_first, iteration_find_path,
         path_length_first, path_length_final, first_path, final_best_path) = result

        if not all(first_path_found):
            continue

        success_count += 1

        # ---- 首次指标 ----
        max_t_first = max(time_first)                # 首次所需时间（所有航段中的最大值）
        sum_len_first = sum(path_length_first)       # 首次路径总长度
        smooth_first = calculate_path_smoothness(first_path)

        # ---- 迭代轮数（首次）----
        sum_iter = sum(iteration_find_path)

        # ---- 最终指标 ----
        total_time = t_end - t_start
        sum_len_final = sum(path_length_final)       # 最终路径总长度
        smooth_final = calculate_path_smoothness(final_best_path)  # final_best_path已合并

        # 存储
        records['max_time_first'].append(max_t_first)
        records['len_first'].append(sum_len_first)
        records['smooth_first'].append(smooth_first)
        records['sum_iter'].append(sum_iter)
        records['time_final'].append(total_time)
        records['len_final'].append(sum_len_final)
        records['smooth_final'].append(smooth_final)

    # 计算均值和标准差
    # def mean_std(lst):
    #     if len(lst) == 0:
    #         return None, None
    #     arr = np.array(lst)
    #     return np.mean(arr), np.std(arr)
    
    def mean_std(lst, lower=5, upper=95):
        if len(lst) == 0:
            return None, None
        arr = np.array(lst)
        mean_all = np.mean(arr)

        # 计算裁剪后的标准差
        arr_sorted = np.sort(arr)
        n = len(arr_sorted)
        low_idx = int(np.floor(n * lower / 100.0))
        high_idx = int(np.ceil(n * upper / 100.0))
        low_idx = max(0, low_idx)
        high_idx = min(n, high_idx)

        if low_idx >= high_idx or high_idx - low_idx < 2:
            # 裁剪后数据太少，退回使用全部数据的标准差
            std_trimmed = np.std(arr)
        else:
            trimmed = arr_sorted[low_idx:high_idx]
            std_trimmed = np.std(trimmed)

        return mean_all, std_trimmed

    stats = {}
    for key in records:
        stats[key] = mean_std(records[key])

    return (step_length, success_count,
            stats['max_time_first'], stats['len_first'], stats['smooth_first'],
            stats['sum_iter'], stats['time_final'], stats['len_final'], stats['smooth_final'])


if __name__ == '__main__':
    # planner_type = 'VRRT'
    # planner_type = 'VRRT_star'
    # planner_type = 'VRRT_star_APF'
    planner_type = 'VRRT_star_Bi'
    # planner_type = 'VRRT_star_Bi_Informed'
    
    # map_type = 'homogeneous'
    map_type = 'cluster'
    # map_type = 'maze'

    # 地图生成
    if map_type == 'homogeneous':
        env_map = env_generator(rho=0.4, 
                                map_dim=(1500, 1500, 240),
                                r_crash_range=(30, 50), 
                                r_risk_range=(3, 7),
                                zmax_range=(30, 240), 
                                max_iter=10000, 
                                seed=1)
    elif map_type == 'cluster':
        env_map = env_generator_cluster(map_dim=(1500, 1500, 240), 
                                        num_clusters=20,
                                        chain_length_range=(1, 4), 
                                        r_center_range=(100, 150),
                                        r_edge_range=(20, 50), 
                                        r_risk_range=(10, 20),
                                        zmax_range=(240, 240), 
                                        min_center_dist=200, 
                                        seed=7)
    elif map_type == 'maze':
        env_map = env_generator_maze(grid_size=(4, 4), 
                                     map_dim=(1500, 1500, 240),
                                     r_crash_range=(40, 60), 
                                     r_risk_range=(10, 20),
                                     zmax_range=(240, 240), 
                                     seed=42)
    else:
        raise ValueError(f"未知的地图类型: {map_type}")

    waypoints = [[0, 0, 0], [360, 806, 54], [832, 1415, 52], [1109, 1491, 85], [1500, 1500, 100]]
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    num_of_tests = 100
    step_lengths = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    print(f"规划器类型: {planner_type}")
    print(f"地图类型: {map_type}")
    print(f"测试步长: {step_lengths}")
    print(f"每个步长重复规划 {num_of_tests} 次")

    tasks = [(step, env_map, waypoints, r_agent_crash, r_agent_risk, num_of_tests, planner_type)
             for step in step_lengths]

    with Pool(processes=10) as pool:
        results = pool.map(evaluate_step_length, tasks)

    # 解析结果
    steps, success_rates = [], []
    avg_data = {
        'max_time_first': [], 'len_first': [], 'smooth_first': [],
        'sum_iter': [], 'time_final': [], 'len_final': [], 'smooth_final': []
    }
    err_data = {
        'max_time_first': [], 'len_first': [], 'smooth_first': [],
        'sum_iter': [], 'time_final': [], 'len_final': [], 'smooth_final': []
    }

    for res in results:
        (step, succ,
         (m_tf, s_tf), (m_lf, s_lf), (m_sf, s_sf),
         (m_it, s_it), (m_tfn, s_tfn), (m_lfn, s_lfn), (m_sfn, s_sfn)) = res

        steps.append(step)
        sr = succ / num_of_tests * 100
        success_rates.append(sr)

        # 用 0 或 None 填充无成功测试的情况
        avg_data['max_time_first'].append(m_tf if m_tf is not None else 0)
        err_data['max_time_first'].append(s_tf if s_tf is not None else 0)
        avg_data['len_first'].append(m_lf if m_lf is not None else 0)
        err_data['len_first'].append(s_lf if s_lf is not None else 0)
        avg_data['smooth_first'].append(m_sf if m_sf is not None else 0)
        err_data['smooth_first'].append(s_sf if s_sf is not None else 0)
        avg_data['sum_iter'].append(m_it if m_it is not None else 0)
        err_data['sum_iter'].append(s_it if s_it is not None else 0)
        avg_data['time_final'].append(m_tfn if m_tfn is not None else 0)
        err_data['time_final'].append(s_tfn if s_tfn is not None else 0)
        avg_data['len_final'].append(m_lfn if m_lfn is not None else 0)
        err_data['len_final'].append(s_lfn if s_lfn is not None else 0)
        avg_data['smooth_final'].append(m_sfn if m_sfn is not None else 0)
        err_data['smooth_final'].append(s_sfn if s_sfn is not None else 0)

        print(f"步长 {step:3d} | 成功: {succ}/{num_of_tests} | 成功率: {sr:.1f}%")

    # 保存结果图的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'test_results')
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 绘制 2×4 子图 ----------
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    x = np.array(steps)

    # 子图索引与标题
    subplot_info = [
        (0, 0, success_rates, None, 'Success rate (%)'),
        (0, 1, avg_data['max_time_first'], err_data['max_time_first'], 'Time first (s)'),
        (0, 2, avg_data['len_first'], err_data['len_first'], 'Length First'),
        (0, 3, avg_data['smooth_first'], err_data['smooth_first'], 'Smoothness First (deg)'),
        (1, 0, avg_data['sum_iter'], err_data['sum_iter'], 'Iterations Total'),
        (1, 1, avg_data['time_final'], err_data['time_final'], 'Time Final (s)'),
        (1, 2, avg_data['len_final'], err_data['len_final'], 'Length Final'),
        (1, 3, avg_data['smooth_final'], err_data['smooth_final'], 'Smoothness Final (deg)')
    ]

    for row, col, y_avg, y_err, title in subplot_info:
        ax = axes[row][col]
        if y_err is not None:   # 带误差棒
            ax.errorbar(x, y_avg, yerr=y_err, fmt='o-', capsize=5, linewidth=2, markersize=6)
        else:                   # 成功率，无误差棒
            ax.plot(x, y_avg, 'o-', linewidth=2, markersize=6)
        ax.set_title(title)
        ax.grid(True)
        if title.startswith('Success rate'):
            ax.set_ylim(0, 110)

    plt.suptitle(f'Step Length Sweep — {planner_type} — {map_type}', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'test_step_length_{planner_type}_{map_type}.png')
    plt.savefig(save_path, dpi=150)
    plt.show()