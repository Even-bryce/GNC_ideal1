import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process
from env_generator_for_data import env_generator
from planner_VRRT import VRRT
from planner_VRRT_star import VRRT_star
from planner_VRRT_star_APF import VRRT_star_APF


def evaluate_waypoints(args):
    """
    在固定的地图和步长下，对给定的航路点组进行多次规划测试，
    返回 (航路点组索引, 成功次数, 平均迭代次数, 平均耗时, 平均长度)
    """
    waypoints, env_map, step_length, r_crash, r_risk, n_tests, idx = args

    # 为每个进程设置不同的随机种子
    pid = current_process().pid
    np.random.seed((idx * 1000 + pid * 997) % 2**31)

    obstacles = env_map["obstacles"]
    success_count = 0
    sum_iter = 0
    sum_time = 0.0
    sum_length = 0.0

    for test_id in range(n_tests):
        rrt = VRRT_star(
            env_map=env_map,
            waypoints=waypoints,
            R_crash=r_crash,
            R_risk=r_risk,
            obstacle_list=obstacles,
            expand_dis=step_length,
            search_radius=90,
            max_iter=10000
        )
        t_start = time.time()
        result = rrt.planning()
        t_end = time.time()

        if result[0] is None:  # 整体规划失败
            continue

        first_path_found, time_first, iteration_find_path, path_length_list, first_path, _ = result

        if all(first_path_found):
            max_iter_needed = np.max(iteration_find_path)
            total_time = t_end - t_start
            total_length = sum(path_length_list)
            sum_iter += max_iter_needed
            sum_time += total_time
            sum_length += total_length
            success_count += 1

    avg_iter = sum_iter / success_count if success_count > 0 else None
    avg_time = sum_time / success_count if success_count > 0 else None
    avg_length = sum_length / success_count if success_count > 0 else None

    return (idx, success_count, avg_iter, avg_time, avg_length)


if __name__ == '__main__':
    # ------------------ 参数设置 ------------------
    env_map = env_generator(
        rho=0.4,
        map_dim=(1500, 1500, 240),
        r_crash_range=(30, 50),
        r_risk_range=(3, 7),
        zmax_range=(30, 240),
        max_iter=10000,
        seed=2
    )

    # 多组航路点（双重列表）
    waypoints_list = [
        [[0, 0, 0], [1500, 1500, 100]],
        [[0, 0, 0], [900, 620, 0], [1500, 1500, 100]],
        [[0, 0, 0], [850, 550, 0], [1500, 1500, 100]],
        [[0, 0, 0], [800, 400, 0], [1200, 1000, 0], [1500, 1500, 100]],
        [[0, 0, 0], [750, 350, 0], [1100, 900, 0], [1500, 1500, 100]],
        [[0, 0, 0], [600, 300, 0], [900, 620, 0], [1200, 1100, 0], [1500, 1500, 100]]]
    num_waypoint_sets = len(waypoints_list)
    
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    num_of_tests = 3000
    step_length = 30  # 固定步长

    print(f"测试航路点组数量: {num_waypoint_sets}")
    print(f"每组重复规划 {num_of_tests} 次，使用 10 进程并行计算")

    # 构建任务列表（每个任务包含航路点组、索引等信息）
    tasks = [(wp, env_map, step_length, r_agent_crash, r_agent_risk, num_of_tests, idx)
             for idx, wp in enumerate(waypoints_list)]

    with Pool(processes=10) as pool:
        results = pool.map(evaluate_waypoints, tasks)

    # 收集结果
    indices = []
    avg_iters = []
    avg_times = []
    avg_lengths = []
    success_rates = []

    for idx, succ, it, ti, le in results:
        indices.append(idx)
        if succ > 0:
            success_rate = succ / num_of_tests
            avg_iters.append(it)
            avg_times.append(ti)
            avg_lengths.append(le)
            success_rates.append(success_rate)
        else:
            avg_iters.append(None)
            avg_times.append(None)
            avg_lengths.append(None)
            success_rates.append(0.0)

        print(f"航路点组 {idx+1:2d} | 成功: {succ}/{num_of_tests} "
              f"| 成功率: {succ/num_of_tests*100:.1f}% "
              + (f"| 平均时间: {ti:.1f}" if ti is not None else "| 失败"))

    # ------------------ 绘图 ------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 只绘制有效数据
    valid = [it is not None for it in avg_iters]
    x_val = np.array(indices)[valid]
    y_iter = np.array(avg_iters)[valid]
    y_time = np.array(avg_times)[valid]
    y_len = np.array(avg_lengths)[valid]

    # 子图1: 平均迭代次数
    ax = axes[0]
    ax.plot(x_val, y_iter, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Waypoint Set Index')
    ax.set_title('Avg iterations ~ waypoint set')
    ax.set_xticks(x_val)
    ax.set_xticklabels([f'Set {i+1}' for i in x_val])
    ax.grid(True)

    # 子图2: 平均耗时
    ax = axes[1]
    ax.plot(x_val, y_time, 's-', linewidth=2, markersize=6)
    ax.set_xlabel('Waypoint Set Index')
    ax.set_title('Avg time ~ waypoint set')
    ax.set_xticks(x_val)
    ax.set_xticklabels([f'Set {i+1}' for i in x_val])
    ax.grid(True)

    # 子图3: 平均路径长度
    ax = axes[2]
    ax.plot(x_val, y_len, '^-', linewidth=2, markersize=6)
    ax.set_xlabel('Waypoint Set Index')
    ax.set_title('Avg length ~ waypoint set')
    ax.set_xticks(x_val)
    ax.set_xticklabels([f'Set {i+1}' for i in x_val])
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('waypoint_comparison_VRRT_star.png', dpi=150)
    plt.show()