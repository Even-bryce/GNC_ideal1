import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process
import os
from env_generator_for_data import env_generator
from planner_VRRT import VRRT
from planner_VRRT_star import VRRT_star
from planner_VRRT_star_APF import VRRT_star_APF


def evaluate_step_length(args):
    """
    在固定的地图和航路点下，对给定步长进行多次规划测试，
    返回 (步长, 成功次数, 平均迭代次数, 平均耗时, 平均长度)
    """
    step_length, env_map, waypoints, r_crash, r_risk, n_tests = args

    # 为每个进程设置不同的随机种子
    pid = current_process().pid
    np.random.seed((int(step_length * 1000) + pid * 997) % 2**31)

    obstacles = env_map["obstacles"]
    success_count = 0
    sum_iter = 0
    sum_time = 0.0
    sum_length = 0.0

    for test_id in range(n_tests):
        rrt = VRRT_star_APF(
            env_map=env_map,
            waypoints=waypoints,
            R_crash=r_crash,
            R_risk=r_risk,
            obstacle_list=obstacles,
            expand_dis=step_length,
            search_radius=110,
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

    return (step_length, success_count, avg_iter, avg_time, avg_length)


if __name__ == '__main__':
    env_map = env_generator(
        rho=0.4,
        map_dim=(1500, 1500, 240),
        r_crash_range=(30, 50),
        r_risk_range=(3, 7),
        zmax_range=(30, 240),
        max_iter=10000,
        seed=40
    )
    waypoints = [[0, 0, 0],
                 [600, 400, 0],
                 [1000, 1000, 0],
                 [1500, 1500, 100]]
    
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    num_of_tests = 3000

    step_lengths = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    print(f"测试步长: {step_lengths}")
    print(f"每个步长重复规划 {num_of_tests} 次，使用 10 进程并行计算")

    tasks = [(step, env_map, waypoints, r_agent_crash, r_agent_risk, num_of_tests)
             for step in step_lengths]

    with Pool(processes=10) as pool:
        results = pool.map(evaluate_step_length, tasks)

    steps = []
    avg_iters = []
    avg_times = []
    avg_lengths = []
    success_rates = []

    for step, succ, it, ti, le in results:
        steps.append(step)
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

        print(f"步长 {step:3d} | 成功: {succ}/{num_of_tests} "
              f"| 成功率: {succ/num_of_tests*100:.1f}% "
              f"| 平均迭代: {it:.1f}" if it is not None else "| 失败")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 只绘制有效数据
    valid = [it is not None for it in avg_iters]
    x_val = np.array(steps)[valid]
    y_iter = np.array(avg_iters)[valid]
    y_time = np.array(avg_times)[valid]
    y_len = np.array(avg_lengths)[valid]

    ax = axes[0]
    ax.plot(x_val, y_iter, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('expand_dis')
    ax.set_title('Avg iterations ~ step length')
    ax.grid(True)

    ax = axes[1]
    ax.plot(x_val, y_time, 's-', linewidth=2, markersize=6)
    ax.set_xlabel('expand_dis')
    ax.set_title('Avg time ~ step length')
    ax.grid(True)

    ax = axes[2]
    ax.plot(x_val, y_len, '^-', linewidth=2, markersize=6)
    ax.set_xlabel('expand_dis')
    ax.set_title('Avg length ~ step length')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('step_length_sweep_VRRT*_APF.png', dpi=150)
    plt.show()