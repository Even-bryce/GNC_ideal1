import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from typing import List, Tuple, Dict, Any
from mpl_toolkits.mplot3d import Axes3D
from env_generator import env_generator
from res_show import plot_map, plot_tree_and_path
from planner_Bias_RRT_star import RRTStar, calculate_path_length

def plan_segment(args):
    """单个航段的规划函数"""
    segment_idx, start, goal, r_agent_crash, r_agent_risk, obstacle_list, env_map, segment_params = args
    
    # 从参数列表中获取当前航段的参数
    expand_dis = segment_params.get('expand_dis', 100)
    max_iter = segment_params.get('max_iter', 100000)
    search_radius = segment_params.get('search_radius', 200.0)
    search_until_max_iter = segment_params.get('search_until_max_iter', False)
    
    # 初始化 RRT*
    rrt_star = RRTStar(
        start=start, 
        goal=goal, 
        R_crash=r_agent_crash, 
        R_risk=r_agent_risk, 
        obstacle_list=obstacle_list, 
        rand_area=[0, env_map["size"], env_map["z_size"]], 
        expand_dis=expand_dis,
        max_iter=max_iter,
        search_radius=search_radius,
        search_until_max_iter=search_until_max_iter
    )
    
    start_time = time.time()
    time_first, iteration_find_path, first_path, final_best_path = rrt_star.planning()
    end_time = time.time()
    time_final = end_time - start_time
    
    result = {
        'segment_idx': segment_idx,
        'start': start,
        'goal': goal,
        'time_first': time_first,
        'time_final': time_final,
        'iteration_find_path': iteration_find_path,
        'first_path': first_path,
        'final_best_path': final_best_path,
        'success': final_best_path is not None,
        'rrt_star_instance': rrt_star
    }
    
    if final_best_path:
        result['length_first'] = calculate_path_length(first_path) if first_path else 0
        result['length_final'] = calculate_path_length(final_best_path)
    
    return result

def multi_waypoint_planning(waypoints: List[List[float]], 
                           segment_params_list: List[Dict],
                           obstacle_list: List,
                           env_map: Dict,
                           r_agent_crash: float = 1.2,
                           r_agent_risk: float = 1.7,
                           num_workers: int = None):
    """
    多航路点并行规划
    :waypoints : 航路点列表List[List[float]]，第一个为起点，最后一个为终点
    :segment_params_list : 每个航段的RRT参数列表List[Dict]，长度应为len(waypoints)-1
    :obstacle_list : 障碍物列表
    :env_map : 环境地图
    :r_agent_crash : 碰撞半径
    :r_agent_risk  : 风险半径
    :num_workers : 并行进程数，None表示使用所有可用CPU核心
    
    返回包含规划结果的字典
    """
    # 错误情况
    if len(waypoints) < 2:
        raise ValueError("航路点小于两个")
    if len(segment_params_list) != len(waypoints) - 1:
        raise ValueError(f"RRT*参数组数量与航段数量不匹配")
    
    # 多进程规划
    tasks = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        goal = waypoints[i + 1]
        tasks.append((
            i,
            start,
            goal,
            r_agent_crash,
            r_agent_risk,
            obstacle_list,
            env_map,
            segment_params_list[i]
        ))

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(tasks))
    
    with mp.Pool(processes=num_workers) as pool:
        segment_results = pool.map(plan_segment, tasks)
    
    # 打印结果
    print("-" * 50)
    print(f"{'航段':<4} | {'状态':<4} | {'迭代轮次':<3} | {'规划时间':<4} | {'路径长度':<4}")
    print("-" * 50)
    
    for result in segment_results:
        status = "成功" if result['success'] else "失败"
        iter_val = f"{result.get('iteration_find_path', 'N/A')}"
        time_val = f"{result.get('time_final', 0):.3f}" if result['success'] else "N/A"
        length_val = f"{result.get('length_final', 0):.2f}" if result['success'] else "N/A"
        
        print(f"{result['segment_idx']+1:<6} | {status:<4} | {iter_val:<8} | {time_val:<8} | {length_val:<8}")
    print("-" * 50)
    
    # 合并路径
    total_path, all_success, total_length = merge_paths(segment_results)
    
    return {
        'segment_results': segment_results,
        'total_path': total_path,
        'all_success': all_success,
        'total_length': total_length,
        'total_time': total_time,
        'waypoints': waypoints
    }
    
def merge_paths(segment_results: List[Dict]) -> Tuple[List, bool, float, float]:
    """合并路径"""
    if not segment_results:
        return [], False, 0
    
    total_path = []
    total_length = 0
    all_success = True
    
    for i, result in enumerate(segment_results):
        if not result['success']:
            all_success = False
            break
        path = result['final_best_path']
        
        # 去除第一个路径的起点
        if i > 0 and path:
            total_path.extend(path[1:])
        else:
            total_path.extend(path)
            
        total_length += result.get('length_final', 0)
    
    return total_path, all_success, total_length

if __name__ == '__main__':
    
    # 生成地图
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
    
    # 航路点
    waypoints = [[0, 0, 0],
                 [1500, 1200, 100],
                 [2550, 2350, 100],
                 [3800, 3800, 100],
                 [5000, 5000, 100]]
    # waypoints = [[0, 0, 0],
    #              [1000, 800, 0],
    #              [2000, 1500, 100],
    #              [3000, 2200, 100],
    #              [3750, 2700, 100],
    #              [4300, 4000, 100],
    #              [5000, 5000, 100]]
    
    # 每个航段的参数
    segment_params_list = [
        {'expand_dis': 100, 'max_iter': 10000, 'search_radius': 200.0},
        {'expand_dis': 100, 'max_iter': 10000, 'search_radius': 200.0},
        {'expand_dis': 100, 'max_iter': 10000, 'search_radius': 200.0},
        # {'expand_dis': 100, 'max_iter': 10000, 'search_radius': 200.0},
        # {'expand_dis': 100, 'max_iter': 10000, 'search_radius': 200.0},
        {'expand_dis': 100, 'max_iter': 10000, 'search_radius': 200.0},
    ]
    
    print(f"使用 {len(waypoints) - 1} 个进程对 {len(waypoints) - 1} 个航段进行并行规划")
    
    # 测试次数
    num_of_tests = 200
    success_count = 0
    total_time = 0
    all_results = []
    
    for test_idx in range(num_of_tests):
        print(f"\n测试 #{test_idx + 1}")
        
        # 执行多航路点规划
        start_time = time.time()

        # 尚未完成：航段数量大于可用CPU核心数量时，num_workers参数设为None，且需另行设计分配方式
        result = multi_waypoint_planning(
            waypoints=waypoints,
            segment_params_list=segment_params_list,
            obstacle_list=obstacle_list,
            env_map=env_map,
            num_workers=len(waypoints) - 1
        )
        
        if result['all_success']:
            success_count += 1
        
        all_results.append(result)
        end_time = time.time()
        time_final = end_time - start_time
        print(f"实际用时：{time_final:.3f}秒")
        total_time += time_final
        # 尚未完成：可视化
        # if result['all_success']:
    
    # 统计结果
    print(f"测试次数: {num_of_tests}")
    print(f"成功率: {success_count/num_of_tests*100:.2f}%")
    
    if success_count > 0:
        successful_results = [r for r in all_results if r['all_success']]
        avg_total_time = total_time / success_count
        avg_total_length = np.mean([r['total_length'] for r in successful_results])
        
        print(f"平均总时间: {avg_total_time:.3f}秒")
        print(f"平均总长度: {avg_total_length:.2f}")