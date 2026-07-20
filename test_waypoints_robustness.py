import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process
import os
import sys
from env_generator_for_data import env_generator, env_generator_maze, env_generator_cluster
from planner_VRRT import VRRT
from planner_VRRT_star import VRRT_star
from planner_VRRT_star_APF import VRRT_star_APF
from planner_VRRT_star_Bi import VRRT_star_Bi
from planner_VRRT_star_Bi_Informed import VRRT_star_Bi_Informed

def evaluate_waypoints(args):
    """
    在固定的地图和步长下，对给定的航路点组进行多次规划测试，
    返回 (航路点组索引, 成功次数, 平均迭代次数, 平均耗时, 平均长度)
    """
    waypoints, env_map, step_length, r_crash, r_risk, n_tests, idx, planner_type = args

    # 为每个进程设置不同的随机种子
    pid = current_process().pid
    np.random.seed((idx * 1000 + pid * 997) % 2**31)

    obstacles = env_map["obstacles"]
    success_count = 0
    sum_time = 0.0
    sum_length_first = 0.0
    sum_length_final = 0.0

    # 根据规划器类型选择类
    if planner_type == 'VRRT':
        rrt = VRRT(
                env_map=env_map,
                waypoints=waypoints,
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacles, 
                expand_dis=step_length,
                max_iter=10000
            )
    elif planner_type == 'VRRT_star':
        rrt = VRRT_star(
                env_map=env_map,
                waypoints=waypoints,
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacles, 
                expand_dis=step_length,
                max_iter=2000,
                search_radius=100,
                search_until_max_iter=True
            ) 
    elif planner_type == 'VRRT_star_APF':
        rrt = VRRT_star_APF(
                env_map=env_map,
                waypoints=waypoints,
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacles, 
                expand_dis=step_length,
                max_iter=2000,
                search_radius=100,
                search_until_max_iter=True
            )
    elif planner_type == 'VRRT_star_Bi':
        rrt = VRRT_star_Bi(
                env_map=env_map,
                waypoints=waypoints,
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacles, 
                expand_dis=step_length,
                max_iter=2000,
                search_radius=100,
                search_until_max_iter=True
            )
    elif planner_type == 'VRRT_star_Bi_Informed':
        rrt = VRRT_star_Bi_Informed(
                env_map=env_map,
                waypoints=waypoints,
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacles, 
                expand_dis=step_length,
                max_iter=2000,
                search_radius=100,
                search_until_max_iter=True
            )
    else:
        raise ValueError(f"未知的规划器类型: {planner_type}")

    for test_id in range(n_tests):
        t_start = time.time()
        result = rrt.planning()
        t_end = time.time()

        if result[0] is None:  # 整体规划失败
            continue

        first_path_found, time_first, iteration_find_path, path_length_list, path_length_final, first_path, final_best_path = result

        if all(first_path_found):
            total_time = min(time_first)
            total_length_first = sum(path_length_list)
            total_length_final = sum(path_length_final)
            sum_time += total_time
            sum_length_first += total_length_first
            sum_length_final += total_length_final
            success_count += 1

    avg_time = sum_time / success_count if success_count > 0 else None
    avg_length_first = sum_length_first / success_count if success_count > 0 else None
    avg_length_final = sum_length_final / success_count if success_count > 0 else None

    return (idx, success_count, avg_time, avg_length_first, avg_length_final)


if __name__ == '__main__':
    # planner_type = 'VRRT'
    # planner_type = 'VRRT_star'
    # planner_type = 'VRRT_star_APF'
    planner_type = 'VRRT_star_Bi'
    # planner_type = 'VRRT_star_Bi_Informed'
    
    # map_type = 'homogeneous'
    # map_type = 'cluster'
    map_type = 'maze'
    
    if map_type == 'homogeneous':
        env_map = env_generator(
            rho=0.4,
            map_dim=(1500, 1500, 240),
            r_crash_range=(30, 50),
            r_risk_range=(3, 7),
            zmax_range=(30, 240),
            max_iter=10000,
            seed=1
        )
    elif map_type == 'cluster':
        env_map = env_generator_cluster(
            map_dim=(1500, 1500, 240),   # (Lx, Ly, Lz)
            num_clusters=20,             # 建议 10~15 之间，保证有足够空间
            chain_length_range=(1, 4),   # 每个簇的圆柱体数量
            r_center_range=(100, 150),    # 接近地图中心的圆柱体半径范围
            r_edge_range=(20, 50),       # 接近地图边缘的圆柱体半径范围
            r_risk_range=(10, 20),       # 风险半径偏移量
            zmax_range=(240, 240),
            min_center_dist=200,         # 【核心参数】任意两个簇中心点的最小绝对距离！
            seed=7,
        )
    elif map_type == 'maze':
        env_map = env_generator_maze(
            grid_size=(4, 4),           # 4x4的网格，网格越多通道越窄越复杂
            map_dim=(1500, 1500, 240),
            r_crash_range=(40, 60),     # 为了给通道留出足够空间，半径相较于你原来设定的(80,125)稍微缩小了一些
            r_risk_range=(10, 20),
            zmax_range=(240, 240),
            seed=42
        )
    else:
        raise ValueError(f"未知的地图类型: {map_type}")

    # 多组航路点
    waypoints_list = [
        [[0, 200, 0], [551, 362, 33], [1136, 695, 36], [1072, 1163, 73], [1500, 1300, 100]],
        # 
        [[0, 200, 0], [553, 376, 44], [1137, 702, 56], [1052, 1141, 86], [1500, 1300, 100]],
        [[0, 200, 0], [558, 388, 57], [1108, 658, 20], [1070, 1149, 97], [1500, 1300, 100]],
        [[0, 200, 0], [584, 360, 18], [1128, 677, 0], [1073, 1168, 119], [1500, 1300, 100]],
        [[0, 200, 0], [549, 383, 60], [1165, 666, 63], [1063, 1170, 31], [1500, 1300, 100]],
        [[0, 200, 0], [548, 389, 61], [1122, 694, 84], [1085, 1174, 66], [1500, 1300, 100]],
        [[0, 200, 0], [543, 366, 78], [1172, 694, 33], [1073, 1162, 122], [1500, 1300, 100]],
        [[0, 200, 0], [532, 322, 28], [1129, 686, 66], [1067, 1165, 36], [1500, 1300, 100]],
        [[0, 200, 0], [547, 410, 33], [1146, 689, 10], [1052, 1139, 62], [1500, 1300, 100]],
        [[0, 200, 0], [553, 336, 30], [1136, 695, 82], [1080, 1164, 40], [1500, 1300, 100]],
        [[0, 200, 0], [570, 328, 60], [1138, 693, 17], [1067, 1163, 95], [1500, 1300, 100]],
        [[0, 200, 0], [553, 344, 75], [1109, 686, 62], [1046, 1177, 111], [1500, 1300, 100]],
        [[0, 200, 0], [548, 338, 66], [1166, 694, 21], [1058, 1161, 108], [1500, 1300, 100]],
        [[0, 200, 0], [536, 387, 25], [1137, 702, 66], [1074, 1175, 32], [1500, 1300, 100]],
        [[0, 200, 0], [580, 376, 68], [1092, 674, 40], [1052, 1172, 78], [1500, 1300, 100]],
        [[0, 200, 0], [547, 329, 13], [1175, 722, 51], [1046, 1172, 80], [1500, 1300, 100]],
        [[0, 200, 0], [551, 370, 1], [1160, 666, 30], [1055, 1165, 95], [1500, 1300, 100]],
        [[0, 200, 0], [583, 374, 47], [1133, 726, -3], [1067, 1165, 35], [1500, 1300, 100]],
        [[0, 200, 0], [515, 349, 1], [1135, 698, 9], [1043, 1155, 44], [1500, 1300, 100]],
        [[0, 200, 0], [534, 352, 33], [1139, 700, -11], [1080, 1182, 67], [1500, 1300, 100]]
    ]
    num_waypoint_sets = len(waypoints_list)
    
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    num_of_tests = 20
    step_length = 10

    print(f"规划器类型: {planner_type}")
    print(f"地图类型: {map_type}")
    print(f"测试航路点组数量: {num_waypoint_sets}")
    print(f"每组重复规划 {num_of_tests} 次，使用 10 进程并行计算")

    # 构建任务列表（每个任务包含航路点组、索引等信息，末尾加入 planner_type）
    tasks = [(wp, env_map, step_length, r_agent_crash, r_agent_risk, num_of_tests, idx, planner_type)
             for idx, wp in enumerate(waypoints_list)]

    with Pool(processes=10) as pool:
        results = pool.map(evaluate_waypoints, tasks)

    # 收集结果
    indices = []
    avg_times = []
    avg_lengths_first = []
    avg_lengths_final = []
    success_rates = []

    for idx, succ, ti, le_first, le_final in results:
        indices.append(idx)
        if succ > 0:
            success_rate = succ / num_of_tests
            avg_times.append(ti)
            avg_lengths_first.append(le_first)
            avg_lengths_final.append(le_final)
            success_rates.append(success_rate)
        else:
            avg_times.append(None)
            avg_lengths_first.append(None)
            avg_lengths_final.append(None)
            success_rates.append(0.0)

        print(f"航路点组 {idx+1:2d} | 成功: {succ}/{num_of_tests} "
              f"| 成功率: {succ/num_of_tests*100:.1f}% "
              + (f"| 平均时间: {ti:.2f}" if ti is not None else "| 失败"))

    # 检查基准组（组1）是否有效
    if avg_times[0] is None or avg_lengths_first[0] is None or avg_lengths_final[0] is None:
        raise ValueError("基准航路点组1没有成功规划，无法进行相对比较。")

    base_time = avg_times[0]
    base_len_first = avg_lengths_first[0]
    base_len_final = avg_lengths_final[0]

    # 计算相对偏差（百分比），仅对有效组
    rel_times = []
    rel_lens_first = []
    rel_lens_final = []
    for i in range(1, len(avg_times)):
        t = avg_times[i]
        lf = avg_lengths_first[i]
        lfl = avg_lengths_final[i]
        if t is not None:
            rel_times.append(((t - base_time) / base_time) * 100)
        if lf is not None:
            rel_lens_first.append(((lf - base_len_first) / base_len_first) * 100)
        if lfl is not None:
            rel_lens_final.append(((lfl - base_len_final) / base_len_final) * 100)

    # 创建保存目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'test_results')
    os.makedirs(save_dir, exist_ok=True)

    # 绘图 —— 三张子图，每张展示相对基准的垂直分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        (axes[0], rel_times, 'Avg First Time', base_time, 's'),
        (axes[1], rel_lens_first, 'Avg First Length', base_len_first, 'm'),
        (axes[2], rel_lens_final, 'Avg Final Length', base_len_final, 'm'),
    ]

    for ax, rel_vals, title, base_val, unit in metrics:
        if not rel_vals:
            ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # 基准线（y=0）
        ax.axhline(0, color='gray', linewidth=1, linestyle='--', label=f'base: {base_val:.3f} {unit}')

        # 所有相对偏差数据点排在 x=1 的垂直线上
        x = np.ones_like(rel_vals)
        # 添加少量横向抖动避免完全重叠（可选）
        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(rel_vals))
        ax.scatter(x + jitter, rel_vals, color='steelblue', alpha=0.7, s=40, zorder=3)

        # 误差棒形式：从最小值到最大值的垂直线段，加两端横线
        y_min = np.min(rel_vals)
        y_max = np.max(rel_vals)
        ax.plot([1, 1], [y_min, y_max], color='darkred', linewidth=2, zorder=2)
        ax.plot([0.9, 1.1], [y_min, y_min], color='darkred', linewidth=2)
        ax.plot([0.9, 1.1], [y_max, y_max], color='darkred', linewidth=2)

        # 标注最大和最小相对偏差
        ax.annotate(f'{y_max:+.1f}%', xy=(1, y_max), xytext=(1.3, y_max),
                    arrowprops=dict(arrowstyle='->', color='darkred'),
                    fontsize=9, color='darkred', va='center')
        ax.annotate(f'{y_min:+.1f}%', xy=(1, y_min), xytext=(1.3, y_min),
                    arrowprops=dict(arrowstyle='->', color='darkred'),
                    fontsize=9, color='darkred', va='center')

        # 美化坐标轴
        ax.set_xlim(0.5, 1.8)  # 留出标注空间
        ax.set_xticks([1])
        ax.set_xticklabels(['Set 2-20'])
        ax.set_ylabel('Relative Error (%)')
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    plt.suptitle(f'Waypoint Robustness Test — {planner_type}')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'test_waypoint_robustness_{planner_type}_{map_type}.png')
    plt.savefig(save_path, dpi=150)
    print(f"图片已保存至: {save_path}")
    plt.show()