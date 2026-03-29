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
        # 去除终点，添加新维度得 [[[x1,y1,z1]],[[x2,y2,z2]],...]
        node_array = waypoints_array[:-1, np.newaxis, :]
        # 创建起点Node类列表[[[start]], [[start*]], [[start**]], ...]
        nodes_list = [[Node(coord[0], coord[1], coord[2])] for coord in node_array[:, 0, :]]
        
        num_trees = len(nodes_list)
        # 终点节点列表
        goal_nodes_list = []
        for i in range(1, num_trees + 1):
            node = Node(waypoints_array[i, 0], waypoints_array[i, 1], waypoints_array[i, 2])
            node.parent = None
            node.cost = float('inf')
            goal_nodes_list.append(node)
            
        first_path_found = np.full(num_trees, False, dtype=bool)
        first_path = np.full(num_trees, None, dtype=object)
        iteration_find_path = np.zeros(num_trees, dtype=int)
        time_first_list = [None] * num_trees
        path_length_list = [None] * num_trees
        
        start_time = time.time()
        
        for i in range(self.max_iter):  # 循环执行最大迭代次数
            node_array = self.build_node_array(nodes_list)
            # 随机采样
            random_nodes_array = self.bias_sample_vectorized(waypoints_array)

            # 找到距离随机点最近的已有节点
            nearest_ind = self.get_nearest_node_index(node_array, random_nodes_array)
            nearest_nodes_list = [nodes_list[i][nearest_ind[i]] for i in range(len(nearest_ind))]
            
            # 计算扩展方向并生成新节点
            new_nodes_list = self.steer(nearest_nodes_list, random_nodes_array)
            
            # 碰撞检测
            collision_results = self.check_collision_vectorized(nearest_nodes_list, new_nodes_list)
            
            # 判断新节点是否可以直连终点
            goal_collision_results = self.check_collision_vectorized(new_nodes_list, goal_nodes_list)
            
            # 检查新节点是否与障碍物碰撞
            for j, node in enumerate(new_nodes_list):
                # 无碰撞，将新节点加入树
                if collision_results[j]:
                    nodes_list[j].append(node)
                    
                # 未找到路径时
                if not first_path_found[j]:
                    
                    # 若存在可直连的节点
                    if goal_collision_results[j] and collision_results[j]:
                        first_path_found[j] = True
                        elapsed = time.time() - start_time
                        time_first_list[j] = elapsed
                        # 生成路径
                        first_path[j] = self.generate_final_path_from_node(new_nodes_list[j])
                        # 补充终点
                        first_path[j].append([goal_nodes_list[j].x, goal_nodes_list[j].y, goal_nodes_list[j].z])
                        path_length_list[j] = calculate_path_length(first_path[j])
                        
                        # 首次找到路径的迭代轮数与时间
                        iteration_find_path[j] = i
                        end_time = time.time()
                        time_first = end_time - start_time
                        
                        if all(first_path_found):
                            # 合并所有航路段
                            combined_path = []
                            for seg in first_path:
                                if combined_path and combined_path[-1] == seg[0]:
                                    combined_path.extend(seg[1:])
                                else:
                                    combined_path.extend(seg)
                                    
                            # 找到首次路径就停止
                            if not self.search_until_max_iter:
                                return first_path_found, time_first_list, iteration_find_path, path_length_list, combined_path, combined_path

        last_index = None
        if last_index is not None:
            # 用剩余迭代次数优化后的路径
            final_best_path = self.generate_final_path_from_node(last_index)
            # 补充终点
            final_best_path.append([self.goal.x, self.goal.y, self.goal.z])
            return time_first, iteration_find_path, first_path, final_best_path 

        return None, None, None, None
    
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
    
    def bias_sample_vectorized(self, waypoints_array, bias_prob = 0.10):
        """
        向量化 bias-RRT 采样
        waypoints: 航路点坐标数组 [[x1,y1,z1], [x2,y2,z2], ..., [xN,yN,zN]]
        bias_prob: 以终点作为采样点的概率
        return: N-1 个采样点坐标数组
        """
        starts = waypoints_array[:-1]
        ends = waypoints_array[1:]

        # 每段航路的随机点生成范围
        mins = np.minimum(starts, ends)
        maxs = np.maximum(starts, ends)

        n_segments = len(starts)
        rng = np.random.default_rng()

        # 随机采样点
        rand = rng.random((n_segments, 4))
        uniform_samples = mins + rand[:, :3] * (maxs - mins)

        # bias掩码
        mask = rand[:, 3] < bias_prob

        # 根据掩码选择终点或采样点
        samples = np.where(mask[:, np.newaxis], ends, uniform_samples)
        
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
        dx = to_nodes_array[:, 0] - from_coords[:, 0]
        dy = to_nodes_array[:, 1] - from_coords[:, 1]
        dz = to_nodes_array[:, 2] - from_coords[:, 2]
        theta = np.arctan2(dy, dx)
        phi = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        
        new_x = from_coords[:, 0] + self.expand_dis * np.cos(theta) * np.cos(phi)
        new_y = from_coords[:, 1] + self.expand_dis * np.sin(theta) * np.cos(phi)
        new_z = from_coords[:, 2] + self.expand_dis * np.sin(phi)

        new_nodes = []
        for i in range(len(from_nodes)):
            new_node = Node(new_x[i], new_y[i], new_z[i])
            new_node.parent = from_nodes[i]
            new_node.cost = from_nodes[i].cost + self.expand_dis + self.risk_cost(new_node)
            new_nodes.append(new_node)
        return new_nodes
    
    def check_collision_vectorized(self, nearest_nodes_lists, new_nodes_list, m = 50):
        """
        向量化碰撞检测函数
        :param nearest_nodes_lists: (N-1,)的节点列表
        :param new_nodes_list: (N-1,)的节点列表
        :param m: 每条线段的采样点数量
        :return: (N-1,)的结果列表，True为无碰撞，False为有碰撞
        """
        N_minus_1 = len(nearest_nodes_lists)
        
        # 转换为 (N-1, 3) 数组
        nearest_coords = np.array([[node.x, node.y, node.z] for node in nearest_nodes_lists])
        new_coords = np.array([[node.x, node.y, node.z] for node in new_nodes_list])
        
        # 在连线上均匀采样 m 个点，包含首尾共 m+2 个检测点
        t_values = np.linspace(0, 1, m + 2)

        # 插值: (1-t) * nearest + t * new，得到 (N-1, m+2, 3) 的采样点
        t_expanded = t_values[np.newaxis, :, np.newaxis]
        sample_points = (1 - t_expanded) * nearest_coords[:, np.newaxis, :] + t_expanded * new_coords[:, np.newaxis, :]
        
        # 展平为 ((N-1)*(m+2), 3) = (N_samples, 3)
        flat_samples = sample_points.reshape(-1, 3)

        # 障碍物参数： (num_obstacles,)
        obstacle_array = np.array(self.obstacle_list)
        obs_x = obstacle_array[:, 0]
        obs_y = obstacle_array[:, 1]
        obs_zmin = obstacle_array[:, 2]
        obs_zmax = obstacle_array[:, 3]
        obs_radius = obstacle_array[:, 4]

        # 每个采样点到每个障碍物中心的水平距离： (N_samples, num_obstacles)
        dx = flat_samples[:, 0, np.newaxis] - obs_x[np.newaxis, :]
        dy = flat_samples[:, 1, np.newaxis] - obs_y[np.newaxis, :]
        horizontal_dist_sq = dx**2 + dy**2
        
        # 比较水平距离与碰撞半径: (N_samples, num_obstacles)
        in_horizontal = horizontal_dist_sq <= (obs_radius[np.newaxis, :]**2)
        
        # 比较采样点纵坐标与垂直高度: (N_samples, num_obstacles)
        z = flat_samples[:, 2, np.newaxis]
        in_vertical = (z >= obs_zmin[np.newaxis, :]) & (z <= obs_zmax[np.newaxis, :])
        
        # 每个采样点在每个障碍物内: (N_samples, num_obstacles)
        in_obstacle = in_horizontal & in_vertical
        
        # 每个采样点在任意障碍物内: (N_samples,)
        any_obstacle = np.any(in_obstacle, axis=1)
        
        # 重塑为 (N-1, m+2)
        collision_by_pair = any_obstacle.reshape(N_minus_1, m + 2)
        
        # 每对节点是否有碰撞：(N-1,)
        no_collision = ~np.any(collision_by_pair, axis=1)
        
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
    num_of_tests = 500  
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
        first_path_found, time_first, iteration_find_path, path_length_list, first_path, final_best_path = rrt_star.planning()
        end_time = time.time()
        
        # 打印单次结果
        num_segments = len(first_path_found)
        print("-" * 50)
        print(f"{'航段':<4} | {'状态':<4} | {'迭代轮次':<3} | {'规划时间':<4} | {'路径长度':<4}")
        print("-" * 50)
        for i in range(num_segments):
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
