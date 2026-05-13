import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from env_generator_for_data import env_generator   # 根据实际路径调整
from planner_VRRT_star import VRRT_star

# ==================== 并行任务函数 ====================
def process_seed(args):
    """
    单个种子的规划任务（独立函数，可被 pickle 序列化）
    参数打包为元组 (seed, env_params, waypoints, r_agent_crash, r_agent_risk, expand_dis, max_iter_rrt, num_tests)
    返回 (seed, avg_time, avg_iter, avg_length, success_count)
    """
    (seed, env_params, waypoints, r_agent_crash, r_agent_risk,
     expand_dis, max_iter_rrt, num_tests) = args

    # 可选：在每个进程中设置不同的随机种子，避免所有进程使用相同的随机序列
    np.random.seed((seed * 1000 + 1) ^ (os.getpid() * 997))

    # 生成地图（每个进程独立生成，保证基于同一 seed 的地图相同）
    env_map = env_generator(seed=seed, **env_params)
    obstacles = env_map["obstacles"]

    sum_time = 0.0
    sum_iter = 0
    sum_length = 0.0
    success_count = 0

    for _ in range(num_tests):
        rrt = VRRT_star(env_map=env_map, waypoints=waypoints,
                  R_crash=r_agent_crash, R_risk=r_agent_risk,
                  obstacle_list=obstacles, expand_dis=expand_dis,
                  max_iter=max_iter_rrt)

        start_t = time.time()
        result = rrt.planning()
        end_t = time.time()

        if result[0] is None:          # 规划失败
            continue

        first_path_found, time_first, iteration_find_path, path_length_list, first_path, _ = result

        if all(first_path_found):      # 所有航段均成功
            max_iter_needed = np.max(iteration_find_path)
            total_time = end_t - start_t
            total_length = sum(path_length_list)

            sum_time += total_time
            sum_iter += max_iter_needed
            sum_length += total_length
            success_count += 1

    if success_count > 0:
        avg_t = sum_time / success_count
        avg_iter = sum_iter / success_count
        avg_len = sum_length / success_count
    else:
        avg_t = None
        avg_iter = None
        avg_len = None

    return (seed, avg_t, avg_iter, avg_len, success_count)


# ==================== 主程序 ====================
if __name__ == '__main__':
    import os  # 用于 getpid

    # 环境参数
    env_params = {
        "rho": 0.4,
        "map_dim": (1500, 1500, 240),
        "r_crash_range": (30, 50),
        "r_risk_range": (3, 7),
        "zmax_range": (30, 240),
        "max_iter": 10000
    }

    # 航路点
    waypoints = [[0, 0, 0],
                 [900, 600, 0],
                 [1500, 1500, 100]]

    # RRT* 参数
    r_agent_crash = 1.2
    r_agent_risk  = 1.7
    expand_dis    = 30
    max_iter_rrt  = 10000

    # 安全种子（硬编码或通过 find_safe_seeds 获得）
    safe_seeds = [0, 2, 9, 10, 11, 13, 15, 16, 17, 18,
                  19, 24, 26, 29, 30, 31, 32, 33, 34, 35,
                  37, 38, 40, 41, 44, 47, 48, 53, 55, 56,
                  57, 58, 59, 61, 63, 64, 66, 69, 70, 78,
                  80, 81, 87, 88, 91, 92, 95, 97, 99, 101]
    num_tests_per_seed = 1000   # 每个种子的重复规划次数

    print(f"找到 {len(safe_seeds)} 个安全种子，使用 10 进程并行规划...")

    # 准备任务参数列表
    tasks = [(seed, env_params, waypoints, r_agent_crash, r_agent_risk,
              expand_dis, max_iter_rrt, num_tests_per_seed) for seed in safe_seeds]

    # 使用 10 进程池并行执行
    with Pool(processes=10) as pool:
        # imap_unordered 可以尽快获取结果（不严格按输入顺序），但我们需要最后按种子排序
        results = list(pool.imap_unordered(process_seed, tasks))

    # 按种子编号排序，然后提取指标
    results.sort(key=lambda x: x[0])   # 按 seed 排序
    avg_total_times = [r[1] for r in results]   # 可能为 None
    avg_total_iters = [r[2] for r in results]
    avg_total_lengths = [r[3] for r in results]
    success_counts = [r[4] for r in results]

    # 打印每个种子的结果
    for i, (seed, avg_t, avg_iter, avg_len, succ) in enumerate(results):
        status = f"成功:{succ}/{num_tests_per_seed}" if succ > 0 else "失败"
        print(f"种子 {seed:3d} ({i+1}/{len(safe_seeds)}) "
              f"{status} "
              f"平均耗时: {avg_t:.3f}s, 平均迭代: {avg_iter:.1f}, 平均长度: {avg_len:.1f}"
              if avg_t is not None else f"种子 {seed:3d} 全部规划失败")

    # 绘图
    x = np.arange(1, len(safe_seeds) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 过滤掉 None 的点，绘图时显示为空白
    valid_mask = [t is not None for t in avg_total_times]
    x_valid = x[valid_mask]
    avg_t_valid = [avg_total_times[i] for i in range(len(x)) if valid_mask[i]]
    avg_iter_valid = [avg_total_iters[i] for i in range(len(x)) if valid_mask[i]]
    avg_len_valid = [avg_total_lengths[i] for i in range(len(x)) if valid_mask[i]]

    ax = axes[0]
    ax.scatter(x_valid, avg_t_valid, s=10, alpha=0.8)
    ax.set_xlabel('Seed index')
    ax.set_title('Avg planning time per seed')
    ax.grid(True)

    ax = axes[1]
    ax.scatter(x_valid, avg_iter_valid, s=10, alpha=0.8)
    ax.set_xlabel('Seed index')
    ax.set_title('Avg iterations per seed')
    ax.grid(True)

    ax = axes[2]
    ax.scatter(x_valid, avg_len_valid, s=10, alpha=0.8)
    ax.set_xlabel('Seed index')
    ax.set_title('Avg path length per seed')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('seed_scatter.png', dpi=150)
    plt.show()