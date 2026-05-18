import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from datetime import datetime
from env_generator_for_data import env_generator
from planner_VRRT import VRRT
from planner_VRRT_star import VRRT_star
from planner_VRRT_star_APF import VRRT_star_APF

def plot_running_averages(data_list, labels, step, planner_name='', figsize=(18, 5),
                           ylim_expand=0.3, save_dir='plots'):
    """
    横向排列绘制多条累积平均值曲线
    ----------
    data_list :  待绘制的指标数据列表
    labels :     每个子图的标题与y轴标签
    step :       取样步长
    planner_name : 规划器名称，用于文件名
    figsize :    图形尺寸
    ylim_expand: 纵轴上下扩展比例
    save_dir :   保存图片的子文件夹名称
    """
    # 获取运行脚本的文件名
    script_path = sys.argv[0]
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.splitext(os.path.basename(script_path))[0]

    # 创建保存目录
    save_path_dir = os.path.join(script_dir, save_dir)
    os.makedirs(save_path_dir, exist_ok=True)

    # 文件名
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = f"{script_name}_{planner_name}_{time_str}.png"

    save_path = os.path.join(save_path_dir, file_name)

    # 绘图
    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, data, label in zip(axes, data_list, labels):
        total = len(data)
        if total < step:
            ax.set_title(f'{label}\n(数据不足{step}次)')
            ax.set_xlabel('Number of Plans')
            ax.set_ylabel(label)
            continue

        n_points = (total // step) * step
        truncated = data[:n_points]

        arr = np.array(truncated)
        cum_avg = np.cumsum(arr) / np.arange(1, n_points + 1)

        x_ticks = np.arange(step, n_points + 1, step)
        y_avg = cum_avg[step - 1::step]

        ax.plot(x_ticks, y_avg, marker='o', linestyle='-')
        ax.set_xlabel('Number of Plans')
        ax.set_title(f'Cumulative Average of {label}')
        ax.grid(True)

        # 放大纵轴范围
        ymin, ymax = np.min(y_avg), np.max(y_avg)
        y_range = ymax - ymin if ymax > ymin else abs(ymin) * 0.1 or 1.0
        ax.set_ylim(ymin - y_range * ylim_expand,
                    ymax + y_range * ylim_expand)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
if __name__ == '__main__':
    # planner_type = 'VRRT'
    planner_type = 'VRRT_star'
    # planner_type = 'VRRT_star_APF'

    # 生成地图
    env_map = env_generator(
        rho=0.4, 
        map_dim=(1500, 1500, 240),
        r_crash_range=(30, 50),
        r_risk_range=(3, 7),
        zmax_range=(30, 240),
        max_iter=10000,
        seed=2
    )
    waypoints = [[0, 0, 0], [650, 380, 0], [1000, 680, 0], [1200, 1100, 0], [1500, 1500, 100]]
    obstacle_list = env_map["obstacles"]
    print(f"地图生成完毕，包含 {len(obstacle_list)} 个障碍物。")
    
    # 设定 RRT* 参数
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    
    # 规划次数
    num_of_tests = 200
    # 测评指标
    success_count = 0
    total_time_first = []      # 首次找到路径的总耗时
    total_iter_needed = []     # 完成规划时，所有航路段的最大迭代次数
    total_length_first = []    # 首次规划的路径总长度
    time_segments = []          # 每个航段的规划时间
    
    for j in range(num_of_tests):
        # 根据 planner_type 实例化对应的规划器
        print(f"\n测试 #{j + 1}  ({planner_type})")
        if planner_type == 'VRRT':
            planner = VRRT(
                env_map=env_map,
                waypoints=waypoints,
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacle_list, 
                expand_dis=20,
                max_iter=10000
            )
        elif planner_type == 'VRRT_star':
            planner = VRRT_star(
                env_map=env_map,
                waypoints=waypoints,
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacle_list, 
                expand_dis=20,
                search_radius=110,
                max_iter=10000
            )
        elif planner_type == 'VRRT_star_APF':
            planner = VRRT_star_APF(
                env_map=env_map,
                waypoints=waypoints,
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacle_list, 
                expand_dis=20,
                search_radius=110,
                max_iter=10000
            )
        else:
            raise ValueError(f"未知的规划器类型: {planner_type}")

        start_time = time.time()
        first_path_found, time_first, iteration_find_path, path_length_list, first_path, final_best_path = planner.planning()
        end_time = time.time()
        
        # 打印单次结果
        num_segments = len(first_path_found)
        print("-" * 50)
        print(f"{'航段':<4} | {'状态':<4} | {'迭代轮次':<3} | {'规划时间':<4} | {'路径长度':<4}")
        print("-" * 50)
        for i in range(num_segments):
            success = first_path_found[i]
            status = "成功" if success else "失败"
            iter_val = str(iteration_find_path[i]) if success else "N/A"
            time_val = f"{time_first[i]:.3f}" if success and time_first[i] is not None else "N/A"
            length_val = f"{path_length_list[i]:.2f}" if success else "N/A"
            print(f"{i+1:<6} | {status:<4} | {iter_val:<8} | {time_val:<8} | {length_val:<8}")
        print("-" * 50)
        
        all_success = all(first_path_found)
        
        if all_success:
            max_iter_needed = np.max(iteration_find_path)
            total_time = end_time - start_time
            total_length = sum([path_length_list[i] for i in range(len(path_length_list)) if first_path_found[i]])
            print(f"总耗时：{total_time:.3f}s")
            print(f"总长度：{total_length:.2f}")
            success_count += 1
            total_time_first.append(total_time)
            total_iter_needed.append(max_iter_needed)
            total_length_first.append(total_length)
            time_segments.append(time_first)
            print(f"当前平均耗时：{sum(total_time_first) / success_count:.3f}s")
            
    print("=" * 30)
    if success_count > 0:
        avg_success_rate = (success_count / num_of_tests) * 100
        avg_time_first = sum(total_time_first) / success_count
        avg_iter = sum(total_iter_needed) / success_count
        avg_length_first = sum(total_length_first) / success_count
        
        print(f"{planner_type}: 步长 {planner.expand_dis}, 搜索半径{planner.search_radius}")
        print(f"成功率: {avg_success_rate:.1f}%")
        print(f"平均迭代次数: {avg_iter:.1f}")
        print(f"平均耗时: {avg_time_first:.4f} s")
        print(f"平均长度: {avg_length_first:.1f}")
    else:
        print("所有任务均失败")
    print("=" * 30)
    
    # 每段航路的平均用时
    if success_count > 0:
        num_segments = len(time_segments[0])
        averages = []
        for i in range(num_segments):
            segment_times = [trial[i] for trial in time_segments]
            avg = sum(segment_times) / len(segment_times)
            averages.append(avg)
        
        print("-" * 20)
        print(f"{'航段':<4} | {'平均时间':<4}")
        print("-" * 20)
        for idx, avg in enumerate(averages, start=1):
            print(f"{idx:<6} | {avg:<8.3f}")
        print("-" * 20)

    plot_running_averages(data_list=[total_time_first, total_iter_needed, total_length_first],
                          labels=['Total Time', 'Iterations', 'Path Length'],
                          step=20,
                          planner_name=planner_type,   # 新增参数
                          ylim_expand=0.3,
                          save_dir='test_results')