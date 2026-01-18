import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import multiprocessing as mp
from typing import List, Tuple, Dict, Any
from mpl_toolkits.mplot3d import Axes3D
from env_generator import env_generator
from res_show import plot_map, plot_tree_and_path
from planner_Bias_RRT_star import RRTStar, calculate_path_length

if __name__ == '__main__':

    env_map = env_generator(
        rho=0.3, 
        map_size=5000,
        r_crash_range=(30, 50),
        r_risk_range=(3, 7),
        zmax_range=(30, 240),
        z_size=240,
        max_iter=10000,
        seed=40
    )
    obstacle_list = env_map["obstacles"]
    print(f"地图生成完毕，包含 {len(obstacle_list)} 个障碍物。")
    # plot_map(env_map)
    
    # 生成一定数量的起终点
    num_of_tasks = 1
    tasks = [([0,0,0],[5000, 5000, 100])] # 对角线起终点
    
    # 运行测试
    success_times = []
    path_lengths = []
    
    # 设定 RRT* 参数
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    env_results = []
    print(f"{'Task':<4} | {'Status':<7} | {'Iter':<6} | {'Time_first':<6}  | {'Length_first':<6} | {'Time_final':<6} | {'Length_final':<6}")
    print("-" * 80)
    
    # 对每个起终点，进行num_pf_tests次规划
    num_of_tests = 500
    for i, (start, goal) in enumerate(tasks):
        env_first_times = []
        env_final_times = []
        env_iters = []
        env_first_lengths = []
        env_final_lengths = []
        env_success_count = 0
        for j in range(num_of_tests):
            # 初始化 RRT*
            rrt_star = RRTStar(
                start=start, 
                goal=goal, 
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacle_list, 
                rand_area=[0, env_map["size"], env_map["z_size"]], 
                expand_dis=150,    # 步长
                max_iter=100000,    # 迭代次数
                search_radius=200.0,
                search_until_max_iter=False
            )
            start_time = time.time()
            time_first, iteration_find_path, first_path, final_best_path = rrt_star.planning()
            end_time = time.time()
            
            time_final = end_time - start_time
            
            if final_best_path:
                plen_first = calculate_path_length(first_path)
                plen_final = calculate_path_length(final_best_path)
                env_first_times.append(time_first)
                env_final_times.append(time_final)
                env_iters.append(iteration_find_path)
                env_first_lengths.append(plen_first)
                env_final_lengths.append(plen_final)
                env_success_count += 1
                print(f"{i+1:<4} | {'Success':<7} | {iteration_find_path:<6} | {time_first:<11.4f} | {plen_first:<12.1f} | {time_final:<10.4f} | {plen_final:<10.1f}")
                # plot_tree_and_path(env_map, rrt_star.node_list, final_best_path)
            else:
                print(f"{i+1:<4} | {'Failed':<7} | {'N/A':<6} | {'N/A':<11} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10}")
        
        env_success_rate = env_success_count / num_of_tests * 100
        env_avg_time_first = sum(env_first_times) / len(env_first_times) if env_first_times else 0
        env_avg_time_final = sum(env_final_times) / len(env_final_times) if env_final_times else 0
        env_avg_iters = sum(env_iters) / len(env_iters) if env_iters else 0
        env_avg_length_first = sum(env_first_lengths) / len(env_first_lengths) if env_first_lengths else 0
        env_avg_length_final = sum(env_final_lengths) / len(env_final_lengths) if env_final_lengths else 0
        
        env_results.append({
            'env_id': i+1,
            'success_rate': env_success_rate,
            'avg_time_first': env_avg_time_first,
            'avg_time_final': env_avg_time_final,
            'avg_iters': env_avg_iters,
            'avg_length_first': env_avg_length_first,
            'avg_length_final': env_avg_length_final
        })        
        print("-" * 80)
            
    # 打印平均值
    print("=" * 80)
    print(f"{'Task':<4} | {'Rate(%)':<4} | {'Iter':<6} | {'Time_first':<6}  | {'Length_first':<6} | {'Time_final':<6} | {'Length_final':<6}")
    print("-" * 80)
    for res in env_results:
        print(f"{res['env_id']:<4} | {res['success_rate']:<7.1f} | {res['avg_iters']:<6.1f} | {res['avg_time_first']:<11.4f} | {res['avg_length_first']:<12.1f} | {res['avg_time_final']:<10.4f} | {res['avg_length_final']:<10.1f}")
    print("-" * 80)