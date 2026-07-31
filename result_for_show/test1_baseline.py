import time
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from env_generator_for_data import env_generator, env_generator_maze, env_generator_cluster
from res_show import plot_map_and_waypoint, plot_tree_and_path

from planner_RRT_star_Bi import RRT_star_Bi
from planner_RRT_star_Bi_DBVS_APF_Informed import RRT_star_DBVSB_APF_Informed
from planner_VRRT_star_Bi import VRRT_star_Bi
from planner_VRRT_star_Bi_Bias_APF_Informed import VRRT_star_Bi_Bias_APF_Informed

def calculate_path_smoothness(path):
    """
    计算路径平滑度
    """
    if len(path) < 3:
        return 0.0

    angles_square = []
    for i in range(len(path) - 2):
        p1 = path[i]
        p2 = path[i + 1]
        p3 = path[i + 2]

        # 向量  p1 -> p2
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
        # 向量  p2 -> p3
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # 跳过长度为零的线段
        if norm1 == 0 or norm2 == 0:
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_square = np.arccos(cos_angle) ** 2
        angles_square.append(angle_square)

    if not angles_square:
        return 0.0

    return float(np.degrees(np.mean(angles_square)))

def load_test_case(filename):
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)
    
    env_map = loaded_data["env_map"]
    final_waypoints = loaded_data["final_waypoints"]
    return env_map, final_waypoints

def evaluate_step_length(args):
    step_length, max_iter, search_radius, env_map, waypoints, r_agent_crash, r_agent_risk, n_tests, planner_type = args
    
    obstacles = env_map["obstacles"]

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
        print(test_id + 1)
        np.random.seed(int(step_length * 1000 + test_id * 7) % 2**31)
        # 根据规划器类型实例化
        if planner_type == 'RRT_star_Bi':
            rrt = RRT_star_Bi(env_map=env_map, 
                            start=waypoints[0], 
                            goal=waypoints[-1], 
                            R_crash=r_agent_crash, 
                            R_risk=r_agent_risk,
                            obstacle_list=obstacles, 
                            expand_dis=step_length,
                            max_iter=max_iter, 
                            search_radius=search_radius, 
                            search_until_max_iter=True)
        elif planner_type == 'RRT_star_DBVSB_APF_Informed':
            rrt = RRT_star_DBVSB_APF_Informed(env_map=env_map, 
                            start=waypoints[0], 
                            goal=waypoints[-1], 
                            R_crash=r_agent_crash, 
                            R_risk=r_agent_risk,
                            obstacle_list=obstacles, 
                            expand_dis=step_length,
                            max_iter=max_iter, 
                            search_radius=search_radius, 
                            search_until_max_iter=True)
        elif planner_type == 'VRRT_star_Bi':
            rrt = VRRT_star_Bi(env_map=env_map, 
                            waypoints=waypoints,
                            R_crash=r_agent_crash, 
                            R_risk=r_agent_risk,
                            obstacle_list=obstacles, 
                            expand_dis=step_length,
                            max_iter=max_iter, 
                            search_radius=search_radius, 
                            search_until_max_iter=True)
        elif planner_type == 'VRRT_star_Bi_Bias_APF_Informed':
            rrt = VRRT_star_Bi_Bias_APF_Informed(env_map=env_map, 
                                        waypoints=waypoints,
                                        R_crash=r_agent_crash, 
                                        R_risk=r_agent_risk,
                                        obstacle_list=obstacles, 
                                        expand_dis=step_length,
                                        max_iter=max_iter, 
                                        search_radius=search_radius, 
                                        search_until_max_iter=True)

        else:
            raise ValueError(f"未知的规划器类型: {planner_type}")
        
        t_start = time.time()
        result = rrt.planning()
        t_end = time.time()

        if result[0] is None:
            continue

        # 解包返回值
        (first_path_found, time_first, iteration_find_path,
         path_length_first, path_length_final, first_path, final_best_path) = result

        if not all(first_path_found):
            continue

        success_count += 1

        max_t_first = max(time_first)                # 首次所需时间（所有航段中的最大值）
        sum_len_first = sum(path_length_first)       # 首次路径总长度
        smooth_first = calculate_path_smoothness(first_path)

        sum_iter = sum(iteration_find_path)

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

    def mean_std(lst):
        if len(lst) == 0:
            return None, None
        arr = np.array(lst)
        return np.mean(arr), np.std(arr)
 
    stats = {}
    for key in records:
        stats[key] = mean_std(records[key])

    return (step_length, success_count,
            stats['max_time_first'], stats['len_first'], stats['smooth_first'],
            stats['sum_iter'], stats['time_final'], stats['len_final'], stats['smooth_final'])


if __name__ == '__main__':
    # map_type = 'homogeneous'
    # map_type = 'cluster'
    map_type = 'maze'

    planner_type_list = ['RRT_star_Bi', 'RRT_star_DBVSB_APF_Informed', 'VRRT_star_Bi', 'VRRT_star_Bi_Bias_APF_Informed']
    
    num_of_tests = 2
    step = 15
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    max_iter = 2000
    search_radius = 50
    
    # 地图生成
    if map_type == 'homogeneous':
        env_map, waypoints = load_test_case('test_case_clutter.pkl') 
    elif map_type == 'cluster':
        env_map, waypoints = load_test_case('test_case_cluster.pkl')
    elif map_type == 'maze':
        env_map, waypoints = load_test_case('test_case_maze.pkl')
    else:
        raise ValueError(f"未知的地图类型: {map_type}")

    plot_map_and_waypoint(env_map, waypoints)

    print(f"规划器类型: {planner_type_list}")
    print(f"地图类型: {map_type}")
    print(f"测试步长: {step}")
    print(f"每个步长重复规划 {num_of_tests} 次")

    results = []
    for planner_type in planner_type_list:
        args = (step, max_iter, search_radius, env_map, waypoints, 
                r_agent_crash, r_agent_risk, num_of_tests, planner_type)
        res = evaluate_step_length(args)
        results.append(res)

    # 解析结果
    planners, success_rates = [], []
    avg_data = {
        'max_time_first': [], 'len_first': [], 'smooth_first': [],
        'sum_iter': [], 'time_final': [], 'len_final': [], 'smooth_final': []
    }
    err_data = {
        'max_time_first': [], 'len_first': [], 'smooth_first': [],
        'sum_iter': [], 'time_final': [], 'len_final': [], 'smooth_final': []
    }

    for res in results:
        (_, succ,
         (m_tf, s_tf), (m_lf, s_lf), (m_sf, s_sf),
         (m_it, s_it), (m_tfn, s_tfn), (m_lfn, s_lfn), (m_sfn, s_sfn)) = res

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

        print(f"{planner_type_list[len(success_rates)-1]:32s} | 成功: {succ}/{num_of_tests} | 成功率: {sr:.1f}%")

    # 保存结果图的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'test_results')
    os.makedirs(save_dir, exist_ok=True)

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
    
    # 为每种规划器定义颜色
    colors = ["#ff0000", "#ff7700", "#ffee00", "#183b00ec", "#51ff00", "#00eeff", "#0037ff", 
              "#4c00ff", "#bb00ff", "#fb00ff", "#ff0066", "#4b000253"]

    # 创建图例元素
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=planner_type_list[i])
                       for i in range(len(planner_type_list))]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    x_pos = np.arange(len(planner_type_list))  # 每个规划器一个位置
                    
    for row, col, y_vals, y_err, title in subplot_info:
        ax = axes[row][col]

        # 绘制每个规划器的柱子（各自颜色）
        for i, (val, err) in enumerate(zip(y_vals, y_err if y_err is not None else [None]*len(y_vals))):
            ax.bar(x_pos[i], val, yerr=err, color=colors[i], capsize=5,
                   edgecolor='black', width=0.6)

        # 隐藏横轴规划器名称（用图例代替）
        ax.set_xticks(x_pos)
        ax.set_xticklabels([''] * len(x_pos))

        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 纵轴范围控制
        if title.startswith('Success rate'):
            ax.set_ylim(0, 110)

        elif 'Iterations' in title:
            # 迭代次数保持默认（通常包含0）
            pass
        else:
            # 非成功率、非迭代次数的指标：纵轴下限不强制为0
            data_min = min(y_vals) if y_vals else 0
            if data_min > 0:
                ax.set_ylim(bottom=data_min * 0.9)
                
        if title == 'Length Final':
            # 获取当前纵轴范围，用于计算文本偏移量
            y_min, y_max = ax.get_ylim()
            offset = (y_max - y_min) * 0.02  # 文本在柱顶上方 2% 范围处

            for i, (val_final, val_first) in enumerate(zip(y_vals, avg_data['len_first'])):
                # 避免除以零
                if val_first > 0:
                    reduction = (val_first - val_final) / val_first * 100
                    label = f'-{reduction:.1f}%' if reduction >= 0 else f'+{-reduction:.1f}%'
                else:
                    label = 'N/A'
                ax.text(x_pos[i], val_final + offset, label,
                        ha='center', va='bottom', fontsize=8, color='black',
                        fontweight='bold')
        if title == 'Smoothness Final (deg)':
            # 获取当前纵轴范围，用于计算文本偏移量
            y_min, y_max = ax.get_ylim()
            offset = (y_max - y_min) * 0.02  # 文本在柱顶上方 2% 范围处

            for i, (val_final, val_first) in enumerate(zip(y_vals, avg_data['smooth_first'])):
                # 避免除以零
                if val_first > 0:
                    reduction = (val_first - val_final) / val_first * 100
                    label = f'-{reduction:.1f}%' if reduction >= 0 else f'+{-reduction:.1f}%'
                else:
                    label = 'N/A'
                ax.text(x_pos[i], val_final + offset, label,
                        ha='center', va='bottom', fontsize=8, color='black',
                        fontweight='bold')

    # 添加统一图例（放在整个图的右上角）
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.98, 0.96), ncol=1, fontsize=9,
               title='Planner', title_fontsize=10)

    plt.suptitle(f'Planner Comparison (step={step}) — {map_type} map', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])  # 为图例留出右侧空间
    save_path = os.path.join(save_dir, f'test_comparison_step={step}_{map_type}.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    # 打印结果
    avg_headers = ['Success Rate(%)','Time First(s)','Len First','Smooth First(deg)',
                   'Iter Total','Time Final(s)','Len Final','Smooth Final(deg)']
    std_headers = ['Time First(s)','Len First','Smooth First(deg)',
                   'Iter Total','Time Final(s)','Len Final','Smooth Final(deg)']

    # 平均值列
    avg_columns = [
        success_rates,
        avg_data['max_time_first'],
        avg_data['len_first'],
        avg_data['smooth_first'],
        avg_data['sum_iter'],
        avg_data['time_final'],
        avg_data['len_final'],
        avg_data['smooth_final']
    ]

    # 标准差列（无成功率）
    std_columns = [
        err_data['max_time_first'],
        err_data['len_first'],
        err_data['smooth_first'],
        err_data['sum_iter'],
        err_data['time_final'],
        err_data['len_final'],
        err_data['smooth_final']
    ]

    # 平均值表格
    avg_formats = [
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.3f}",
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.0f}",
        lambda v: f"{v:.3f}",
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.1f}",
    ]

    # 标准差表格：无成功率，其余与平均值对应
    std_formats = [
        lambda v: f"{v:.3f}",
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.0f}",
        lambda v: f"{v:.3f}",
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.1f}",
    ]

    def print_metric_table(title, headers, planners, columns, col_formats):
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
        header_row = f"{'Planner':<32}"
        for h in headers:
            header_row += f"{h:>18}"
        print(header_row)
        print('-' * len(header_row))
        for i, planner in enumerate(planners):
            row = f"{planner:<32}"
            for idx, col in enumerate(columns):
                val = col[i]
                formatted_val = col_formats[idx](val)
                row += f"{formatted_val:>18}"
            print(row)
        print()

    # 分别打印平均值和标准差
    print_metric_table("平均指标 (Average Metrics)", avg_headers, planner_type_list, avg_columns, avg_formats)
    print_metric_table("标准差 (Standard Deviation)", std_headers, planner_type_list, std_columns, std_formats)