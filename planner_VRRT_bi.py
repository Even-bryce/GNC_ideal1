import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from env_generator import env_generator
from res_show import plot_map, plot_tree_and_path

# 定义 Node 类，用于表示树中的每个节点
class Node:
    def __init__(self, x, y, z):
        self.x = x              # 节点的 x 坐标
        self.y = y              # 节点的 y 坐标
        self.z = z              # 节点的 z 坐标
        self.parent = None      # 节点的父节点，用于回溯路径
        self.cost = 0.0         # 从起点到该节点的路径成本

# 定义 RRT 类，用于实现 RRT 算法
class RRT:
    def __init__(self, waypoints, R_crash, R_risk, obstacle_list, expand_dis=25, max_iter=1500, search_until_max_iter=False):
        """
        初始化 RRT 算法的参数
        :param waypoints: 航路点坐标列表 [[x1,y1,z1],[x2,y2,z2],...,[xN,yN,zN]]
        :param obstacle_list: 障碍物列表，每个障碍物为 [x, y, zmin, zmax, R_ob_crash, R_ob_risk]
        :param rand_area: 随机采样区域的范围 [min, max]
        :param expand_dis: 树扩展的步长
        :param max_iter: 最大迭代次数
        :param search_radius: 搜索邻近节点的半径
        :param R_crash: 飞行器碰撞半径
        :param R_risk: 飞行器风险半径
        """
        self.waypoints = waypoints             # 航路点列表
        self.expand_dis = expand_dis           # 每次扩展的步长
        self.max_iter = max_iter               # 最大迭代次数
        self.obstacle_list = obstacle_list     # 存储障碍物列表
        self.node_list = []                    # 树节点列表，初始化为空列表
        self.R_crash = R_crash                 # 本体碰撞半径
        self.R_risk = R_risk                   # 本体风险半径
        self.search_until_max_iter = search_until_max_iter  # 是否持续搜索直到最大迭代次数

    def planning(self):
        """
        主规划函数，用于生成从起点到目标的路径
        返回首次找到的可行路径，和循环结束后找到的最优路径；否则返回 None
        """
        # 转换为 (n,3) 数组
        waypoints_array = np.array(self.waypoints)
        
        num_trees = len(waypoints_array) - 1
        
        # 初始化双向树
        trees_start = []
        trees_goal = []
        for i in range(num_trees):
            start_node = Node(waypoints_array[i,0], waypoints_array[i,1], waypoints_array[i,2])
            goal_node = Node(waypoints_array[i+1,0], waypoints_array[i+1,1], waypoints_array[i+1,2])
            goal_node.parent = None
            trees_start.append([start_node])
            trees_goal.append([goal_node])
        
        path_found = np.full(num_trees, False, dtype=bool)
        final_paths = [None] * num_trees
        iteration_find_path = np.zeros(num_trees, dtype=int)
        time_first_list = [None] * num_trees
        path_length_list = [None] * num_trees
        combined_path = []
        
        start_time = time.time()
        for i in range(self.max_iter):
            start_array = self.build_node_array(trees_start)
            goal_array = self.build_node_array(trees_goal)
            # 随机采样
            random_nodes_array = self.sample_free_vectorized(waypoints_array)
            # 找到最近节点
            nearest_ind_start = self.get_nearest_node_index(start_array, random_nodes_array)
            nearest_nodes_start = [trees_start[i][nearest_ind_start[i]] for i in range(num_trees)]
            # 扩展新节点
            new_nodes_start = self.steer(nearest_nodes_start, random_nodes_array)
            # 碰撞检测
            collision_results_start = self.check_collision_vectorized(nearest_nodes_start, new_nodes_start)
            
            # 连接终点树碰撞检测
            new_nodes_start_array = np.array([[node.x, node.y, node.z] for node in new_nodes_start])
            nearest_ind_connect = self.get_nearest_node_index(goal_array, new_nodes_start_array)
            nearest_nodes_connect = [trees_goal[i][nearest_ind_connect[i]] for i in range(num_trees)]
            collision_finish_results = self.check_collision_vectorized(new_nodes_start, nearest_nodes_connect)
            
            for j, node in enumerate(new_nodes_start):
                # 无碰撞，将新节点加入树
                if collision_results_start[j]:
                    trees_start[j].append(node)
                    
                if not path_found[j]:
                    # 若存在可直连的节点
                    if collision_results_start[j] and collision_finish_results[j]:
                        path_found[j] = True
                        elapsed = time.time() - start_time
                        time_first_list[j] = elapsed
                        # 生成路径
                        start_path = self.generate_final_path_from_node(new_nodes_start[j])          # 从起点到新节点
                        goal_path_rev = self.generate_final_path_from_node(nearest_nodes_connect[j]) # 从终点到连接节点
                        goal_path = list(reversed(goal_path_rev))                              # 从连接节点到终点
                        full_path = start_path + goal_path
                        final_paths[j] = full_path
                        path_length_list[j] = calculate_path_length(final_paths[j])
                        # 首次找到路径的迭代轮数
                        iteration_find_path[j] = i
                        
                        if all(path_found):
                            # 合并所有航路段
                            for seg in final_paths:
                                if combined_path and combined_path[-1] == seg[0]:
                                    combined_path.extend(seg[1:])
                                else:
                                    combined_path.extend(seg)
                            return path_found, time_first_list, iteration_find_path, path_length_list, combined_path
            trees_start, trees_goal = trees_goal, trees_start 
        return path_found, time_first_list, iteration_find_path, path_length_list, None   
    
    def build_node_array(self, nodes_list):
        max_len = max(len(tree) for tree in nodes_list)
        tree_arrays = []
        for tree in nodes_list:
            # 提取当前树所有节点的坐标，形状 (len(tree), 3)
            coords = np.array([[node.x, node.y, node.z] for node in tree])
            # 若节点数不足最大长度，用 np.inf 填充尾部
            if len(tree) < max_len:
                pad = np.full((max_len - len(tree), 3), np.inf)
                coords = np.vstack([coords, pad])
            tree_arrays.append(coords)
        # 堆叠为 (num_trees, max_len, 3)
        return np.array(tree_arrays)
    
    def sample_free_vectorized(self, waypoints_array):
        """
        向量化随机采样
        waypoints: 航路点坐标数组 [[x_1,y_1,z_1],[x_2,y_2,z_2],...,[x_N,y_N,z_N]]
        return: N-1个采样点坐标数组 [rnd_1, rnd_2, ...]
        """
        # (n-1,3) 的起点与终点数组
        starts = waypoints_array[:-1]
        ends = waypoints_array[1:]
        
        # 每段航路的随机点生成范围
        mins = np.minimum(starts, ends)
        maxs = np.maximum(starts, ends)
        
        # 生成 [0,1) 间的 (n-1,3) 随机数组
        n_segments = len(starts)
        
        rnd_gen = np.random.default_rng()
        random_points = rnd_gen.random((n_segments, 3))
        
        # 缩放得到采样点
        samples = mins + random_points * (maxs - mins)
        
        return samples
        
    def sample_goal(self, goal_sample_rate):
        """
        根据给定的采样率决定是否采样目标点
        :param goal_sample_rate: 采样目标点的概率（0-100）
        :return: 采样点的坐标 [x, y, z]
        """
        if random.randint(0, 100) > goal_sample_rate:
            return self.sample_free()
        else:
            return [self.goal.x, self.goal.y, self.goal.z]
    
    def get_nearest_node_index(self, node_array, rnd_array):
        """
        找到N-1棵树中，距离N-1个随机点最近的节点的索引
        :param node_array: N-1棵树的节点坐标数组 (N-1, M, 3)
        :param rnd_array: N-1个采样点坐标数组 (N-1, 3)
        :return: 最近节点的索引
        """
        # 扩展维度至 (N-1, 1, 3)
        rnd_expanded = rnd_array[:, np.newaxis, :]
        
        # 计算 (N-1, M, 3) - (N-1, 1, 3) 的平方和，得距离矩阵 (N-1, M)
        diff = node_array - rnd_expanded
        distances_sq = np.sum(diff ** 2, axis=2)
        
        # 每棵树中距离最小的节点索引 (N-1,)
        min_indices = np.argmin(distances_sq, axis=1)
        
        return min_indices
    
    def steer(self, from_nodes, to_nodes_array):
        """
        从 from_nodes 向 to_nodes 扩展新节点
        :param from_nodes: 起始节点
        :param to_nodes: 目标节点
        :return: 新节点
        """
        from_coords = np.array([[node.x, node.y, node.z] for node in from_nodes])
        dir_vec = to_nodes_array - from_coords
        dist = np.linalg.norm(dir_vec, axis=1, keepdims=True)
        step = np.minimum(self.expand_dis, dist)
        new_coords = from_coords + (dir_vec / dist) * step
        new_nodes = []
        for i in range(len(from_nodes)):
            new_node = Node(new_coords[i, 0], new_coords[i, 1], new_coords[i, 2])
            new_node.parent = from_nodes[i]
            actual_dist = step[i, 0] if dist[i,0] > 0 else 0
            new_node.cost = from_nodes[i].cost + actual_dist
            new_nodes.append(new_node)
        return new_nodes
    
    def check_collision_vectorized(self, nearest_nodes_lists, new_nodes_list, m=50):
        """
        优化版向量化碰撞检测：每个航段只检测其空间范围内的障碍物
        :param nearest_nodes_lists: 起点节点列表 (N-1,)
        :param new_nodes_list: 终点节点列表 (N-1,)
        :param m: 每条线段内部的采样点数（不含端点）
        :return: (N-1,) 布尔列表，True表示无碰撞
        """
        N = len(nearest_nodes_lists)
        
        # 转换为 (N,3) 坐标数组
        starts = np.array([[node.x, node.y, node.z] for node in nearest_nodes_lists])
        ends   = np.array([[node.x, node.y, node.z] for node in new_nodes_list])
        
        # 每个航段的包围盒
        mins = np.minimum(starts, ends)
        maxs = np.maximum(starts, ends)
        
        # 障碍物参数
        obs = np.array(self.obstacle_list)
        obs_x = obs[:, 0]
        obs_y = obs[:, 1]
        obs_zmin = obs[:, 2]
        obs_zmax = obs[:, 3]
        obs_r = obs[:, 4]
        O = len(obs)
        
        # 障碍物包围盒
        obs_xmin = obs_x - obs_r
        obs_xmax = obs_x + obs_r
        obs_ymin = obs_y - obs_r
        obs_ymax = obs_y + obs_r
        
        # 航段与障碍物的包围盒重叠检测（全向量化）
        # 形状 (N, O)
        overlap_x = (maxs[:, 0:1] >= obs_xmin) & (mins[:, 0:1] <= obs_xmax)
        overlap_y = (maxs[:, 1:2] >= obs_ymin) & (mins[:, 1:2] <= obs_ymax)
        overlap_z = (maxs[:, 2:3] >= obs_zmin) & (mins[:, 2:3] <= obs_zmax)
        overlap = overlap_x & overlap_y & overlap_z   # True 表示该航段可能与障碍物碰撞
        
        # 采样参数
        t_vals = np.linspace(0, 1, m + 2)
        t_exp = t_vals[np.newaxis, :, np.newaxis]
        
        # 初始化：无碰撞
        no_collision = np.ones(N, dtype=bool)
        
        # 对每个障碍物单独处理
        for j in range(O):
            # 需要检测该障碍物的航段索引
            seg_idx = np.where(overlap[:, j])[0]
            if len(seg_idx) == 0:
                continue
            
            # 取出这些航段的起终点
            start_j = starts[seg_idx]      # (k,3)
            end_j   = ends[seg_idx]        # (k,3)
            
            # 生成这些航段上的采样点 (k, m+2, 3)
            sample_j = (1 - t_exp) * start_j[:, np.newaxis, :] + t_exp * end_j[:, np.newaxis, :]
            flat_j = sample_j.reshape(-1, 3)   # (k*(m+2), 3)
            
            # 对该障碍物进行碰撞检测
            dx = flat_j[:, 0] - obs_x[j]
            dy = flat_j[:, 1] - obs_y[j]
            dist2_horiz = dx*dx + dy*dy
            in_horiz = dist2_horiz <= obs_r[j]*obs_r[j]
            
            z = flat_j[:, 2]
            in_vert = (z >= obs_zmin[j]) & (z <= obs_zmax[j])
            
            in_obs = in_horiz & in_vert
            
            # 重塑为 (k, m+2)，并判断每个航段是否有碰撞
            coll_j = in_obs.reshape(len(seg_idx), m+2)
            any_coll = np.any(coll_j, axis=1)   # 长度为k的布尔数组
            
            # 更新结果：如果当前障碍物导致某航段碰撞，则标记为False
            no_collision[seg_idx] &= ~any_coll
            
            # 若所有航段均已确定碰撞，可提前退出（可选）
            if not np.any(no_collision):
                break
        
        return no_collision.tolist()
    
    def generate_final_path_from_node(self, end_node):
        path = [[end_node.x, end_node.y, end_node.z]]
        node = end_node
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y, node.z])
        return path[::-1]
    
    # --------------------------工具函数--------------------------
    def risk_cost(self, node):
        """
        计算节点处需付的风险代价（soft penalty）。
        风险区定义为： dist <= (agent_R_risk + obs_R_risk) 并且超出 crash 阈值。
        惩罚随靠近 crash 边界上升。
        """
        penalty = 0.0
        for cx, cy, z_min, z_max, obs_R_crash, obs_R_risk in self.obstacle_list:
            if z_min <= node.z <= z_max:
                dist_xy = math.hypot(node.x - cx, node.y - cy)
                crash_threshold = (self.R_crash + obs_R_crash)
                risk_threshold = (self.R_risk + obs_R_risk)
                # 若在 risk 区间内但不在 crash 内
                if crash_threshold < dist_xy <= risk_threshold:
                    # margin 越小惩罚越大
                    margin = max(1e-3, dist_xy - crash_threshold)
                    penalty += 0.0 / margin  # 权重可调
        return penalty
    
def calculate_path_length(path):
    length = 0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        length += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    return length

# ==========================================
# 4. 主程序
# ==========================================
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
                 [1000, 800, 0],
                 [2000, 1500, 100],
                 [3000, 2200, 100],
                 [3750, 2700, 100],
                 [3800, 3700, 100],
                 [5000, 5000, 100]]
    # waypoints = [[0, 0, 0],
    #              [1500, 1200, 100],
    #              [2550, 2350, 100],
    #              [3800, 3800, 100],
    #              [5000, 5000, 100]]
    
    # 设定 RRT* 参数
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    env_results = []
    
    # 规划次数
    num_of_tests = 1000  
    # 测评指标
    success_count = 0
    total_time_first = []      # 首次找到路径的总耗时
    total_time_final = []      # 最终找到路径的总耗时
    total_iter_needed = []     # 完成规划时，所有航路段的最大迭代次数
    total_length_first = []    # 首次规划的路径总长度
    total_length_final = []    # 最终规划的路径总长度
    
    for j in range(num_of_tests):
        # 初始化 RRT*
        print(f"\n测试 #{j + 1}")
        rrt_star = RRT(
            waypoints=waypoints,
            R_crash=r_agent_crash, 
            R_risk=r_agent_risk, 
            obstacle_list=obstacle_list, 
            expand_dis=100,
            max_iter=1000
        )
        start_time = time.time()
        first_path_found, time_first, iteration_find_path, path_length_list, first_path = rrt_star.planning()
        end_time = time.time()
        
        # 打印单次结果
        num_trees = len(first_path_found)
        print("-" * 50)
        print(f"{'航段':<4} | {'状态':<4} | {'迭代轮次':<3} | {'规划时间':<4} | {'路径长度':<4}")
        print("-" * 50)
        for i in range(num_trees):
            success = first_path_found[i]
            status = "成功" if success else "失败"
            # 迭代轮次
            iter_val = str(iteration_find_path[i]) if success else "N/A"
            # 规划时间
            time_val = f"{time_first[i]:.3f}" if success and time_first[i] is not None else "N/A"
            # 路径长度
            length_val = f"{path_length_list[i]:.2f}" if success else "N/A"
            # 航段编号
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
    
    print("=" * 30)
    if success_count > 0:
        avg_success_rate = (success_count / num_of_tests) * 100
        avg_time_first = sum(total_time_first) / success_count
        avg_iter = sum(total_iter_needed) / success_count
        avg_length_first = sum(total_length_first) / success_count

        print(f"成功率: {avg_success_rate:.1f}%")
        print(f"平均迭代次数: {avg_iter:.1f}")
        print(f"平均耗时: {avg_time_first:.4f} s")
        print(f"平均长度: {avg_length_first:.1f}")
    else:
        print("所有任务均失败")

    print("=" * 30)
    
    if success_count > 0:
    # 设置直方图参数
        bins = 'auto'  # 自动选择合适的分组数
        alpha = 0.7    # 透明度
        color = 'skyblue'
        edgecolor = 'black'

        # 绘制直方图
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(total_time_first, bins=bins, alpha=alpha, 
                                    color=color, edgecolor=edgecolor)

        # 均值线
        avg_time_first = np.mean(total_time_first)
        plt.axvline(avg_time_first, color='red', linestyle='dashed', linewidth=1.5,
                    label=f'average = {avg_time_first:.4f} s')

        # 图表装饰
        plt.xlabel('time (s)', fontsize=12)
        plt.ylabel('number', fontsize=12)
        plt.title('Time of Bi_RRT', fontsize=14)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # 显示图形
        plt.show()
