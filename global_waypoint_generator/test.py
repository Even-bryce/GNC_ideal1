import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from src.data.tools_for_data_generation.env_generator_for_data import env_generator, env_generator_cluster
from src.data.tools_for_data_generation.res_show_for_data import plot_map, plot_tree_and_path_pv
from src.models.pointnet_transfomer2.my_model import get_model
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import torch
# 定义 Node 类，用于表示树中的每个节点
class Node:
    def __init__(self, x, y, z):
        self.x = x              # 节点的 x 坐标
        self.y = y              # 节点的 y 坐标
        self.z = z              # 节点的 z 坐标
        self.parent = None      # 节点的父节点，用于回溯路径
        self.cost = 0.0         # 从起点到该节点的路径成本

# 定义 RRT 类，用于实现 RRT 算法
class RRT_vector:
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
            random_nodes_array = self.bias_sample_vectorized2(waypoints_array)

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

        # 情况 1：如果是因为 search_until_max_iter=True 跑完了全程，且所有路径都找到了
        if all(first_path_found):
            combined_path = []
            for seg in first_path:
                if combined_path and combined_path[-1] == seg[0]:
                    combined_path.extend(seg[1:])
                else:
                    combined_path.extend(seg)
            return first_path_found, time_first_list, iteration_find_path, path_length_list, combined_path, combined_path
            
        # 情况 2：达到了最大迭代次数，但依然有航段没找通（打破木桶效应）
        else:
            # 找出到底是哪几段失败了
            failed_segments = [idx for idx, found in enumerate(first_path_found) if not found]
            print(f"\n[警告] 达到最大迭代次数 ({self.max_iter})，以下航段未能找到路径: {failed_segments}")
            
            # 把已经找到的 partial path 数组直接返回，千万别 return None
            # 这样你在外面调用时，依然能画出部分成功的路径，方便 debug
            return first_path_found, time_first_list, iteration_find_path, path_length_list, first_path, None
        
    
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
    
    def bias_sample_vectorized2(self, waypoints_array, bias_prob=0.20, map_bounds=None, padding=50.0):
        """
        向量化 bias-RRT 全局采样
        :param waypoints_array: 航路点坐标数组 [[x1,y1,z1], [x2,y2,z2], ..., [xN,yN,zN]]
        :param bias_prob: 以终点作为采样点的概率 (默认 10%)
        :param map_bounds: 显式指定的全图边界 [[xmin, xmax], [ymin, ymax], [zmin, zmax]]。如果为None则自动计算。
        :param padding: 自动计算边界时的外扩缓冲距离（米）
        :return: N-1 个采样点坐标数组 (N-1, 3)
        """
        starts = waypoints_array[:-1]
        ends = waypoints_array[1:]
        n_segments = len(starts)

        # ---------------- 确定全局采样边界 ----------------
        if map_bounds is not None:
            # 方式 1：使用用户指定的物理地图边界
            min_bound = np.array([map_bounds[0][0], map_bounds[1][0], map_bounds[2][0]])
            max_bound = np.array([map_bounds[0][1], map_bounds[1][1], map_bounds[2][1]])
        else:
            # 方式 2：根据所有航路点，算出全局 Bounding Box，并向外扩张 padding 距离
            # 注意：这里用的是整个 waypoints_array，而不是单段的 starts/ends
            min_bound = np.min(waypoints_array, axis=0) - padding
            max_bound = np.max(waypoints_array, axis=0) + padding

        rng = np.random.default_rng()
        
        # 生成 (N-1, 4) 的随机数组。前3列用于xyz坐标，第4列用于计算目标偏置(bias)概率
        rand = rng.random((n_segments, 4))

        # 在全局边界内生成均匀分布的随机点
        uniform_samples = min_bound + rand[:, :3] * (max_bound - min_bound)

        # ---------------- 目标偏置 (Goal-Bias) ----------------
        mask = rand[:, 3] < bias_prob

        # 根据掩码，决定是采用全局随机点，还是直接取该段的终点
        # 注意：这里的 ends 依然是各自航段的真实目标，这样能保证引力的方向是正确的
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

class RRTStar:
    def __init__(self, start, goal, R_crash, R_risk, obstacle_list, rand_area, expand_dis=30, max_iter=1500, search_radius=150, search_until_max_iter=True):
        """
        初始化 RRT* 算法的参数
        :param start: 起点坐标 [x, y, z]
        :param goal: 目标坐标 [x, y, z]
        :param obstacle_list: 障碍物列表，每个障碍物为 [x, y, zmin, zmax, R_ob_crash, R_ob_risk]
        :param rand_area: 随机采样区域的范围 [min, max], 大小为2*3的列表
        :param expand_dis: 树扩展的步长
        :param max_iter: 最大迭代次数
        :param search_radius: 搜索邻近节点的半径
        :param R_crash: 飞行器碰撞半径
        :param R_risk: 飞行器风险半径
        """
        self.start = Node(start[0], start[1], start[2])  # 创建起点节点
        self.goal = Node(goal[0], goal[1], goal[2])     # 创建目标节点
        self.min_rand = rand_area[0]          # 随机采样区域的最小值
        self.max_rand = rand_area[1]           # 随机采样区域的最大值
       
        self.expand_dis = expand_dis           # 每次扩展的步长
        self.max_iter = max_iter               # 最大迭代次数
        self.obstacle_list = obstacle_list     # 存储障碍物列表
        self.node_list = [self.start]          # 树节点列表，初始化时只包含起点
        self.search_radius = search_radius     # 搜索邻近节点的半径
        self.R_crash = R_crash   # 本体碰撞半径
        self.R_risk = R_risk     # 本体风险半径
        self.search_until_max_iter = search_until_max_iter  # 是否持续搜索直到最大迭代次数
        self.first_path_found = False  # 是否已找到首条路径
        self.c_best = float("inf")             # 当前最佳路径成本
        self.c_min = self.calc_dist_to_goal(self.start.x, self.start.y, self.start.z)
        self.use_informed_sampling = False     # 是否启用 Informed 采样
        

    def planning(self):
        """
        主规划函数，用于生成从起点到目标的路径
        :return: 如果找到路径，返回路径坐标列表；否则返回 None
        """

        self.goal.cost = float('inf')
        self.goal.parent = None
        start_time = time.time()

        for i in range(self.max_iter):  # 循环执行最大迭代次数
            # 随机采样一个点
            if not self.first_path_found:
                rnd = self.sample_goal(20)
            else:
                rnd = self.informed_sample()
            
            # rnd = self.sample_free()
        
            # 找到距离随机点最近的已有节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            # 计算扩展方向并生成新节点，steer内部会自动计算新节点的成本
            steer_node = self.steer(nearest_node, Node(rnd[0], rnd[1], rnd[2]))
            new_node = self.apf_steer(steer_node, self.goal)
    
            # 检查新节点是否与障碍物碰撞
            if (not self.check_collision(new_node) and 
                    not self.check_edge_collision(nearest_node, new_node)):
                # 找到新节点附近的节点
                near_inds = self.find_near_nodes(new_node)
                # 选择最佳父节点
                node_with_updated_parent = self.choose_parent(new_node, near_inds) # 重新选择父节点
                # 如果父节点更新了
                if node_with_updated_parent:
                    # 重布线
                    self.rewire(node_with_updated_parent, near_inds)
                    # 只有这里才把节点添加到树中
                    self.node_list.append(node_with_updated_parent)
                    
                else:
                    # 重布线
                    self.rewire(new_node, near_inds)
                    # 只有这里才把节点添加到树中
                    self.node_list.append(new_node)
                
                # 记录首次找到路径的信息
                if not self.first_path_found:
                    # 1. 尝试寻找是否能连通终点
                    potential_goal_ind = self.search_best_goal_node()
                    
                    # 如果找到了 (不为 None)
                    if potential_goal_ind is not None:
                        self.first_path_found = True
                        self.use_informed_sampling = True
                        # 计算最短路径下界
                        
                        # 生成坐标路径
                        first_path_coords = self.generate_final_path_from_node(potential_goal_ind)
                        # 补充终点
                        first_path_coords.append([self.goal.x, self.goal.y, self.goal.z])
                        
                        # 计算纯几何长度
                        first_path_len = self.calculate_path_length(first_path_coords)
                        self.c_best = first_path_len
                        end_time = time.time()
                        first_time = end_time - start_time
                        
                        
                        if self.search_until_max_iter:
                            print(f"\n[提示] 发现首条可行路径！迭代次数: {i}")
                            print(f"[数据] 首条路径物理长度: {first_path_len:.4f} 米")
                            print(f"[数据] 首条路径计算时间: {first_time:.4f} 秒")
                        else:
                            print("[提示] 由于设置为不持续搜索，规划结束。\n")
                            return first_path_coords
                
            temp_goal_ind = self.search_best_goal_node()
            if temp_goal_ind is not None:
                temp_path_coords = self.generate_final_path_from_node(temp_goal_ind)
                temp_path_coords.append([self.goal.x, self.goal.y, self.goal.z])
                temp_path_len = self.calculate_path_length(temp_path_coords)
                if temp_path_len < self.c_best:
                    self.c_best = self.node_list[temp_goal_ind].cost + self.calc_distance(self.node_list[temp_goal_ind], self.goal)
                        
        
        last_index = self.search_best_goal_node()

        if last_index is not None:
            path_coords = self.generate_final_path_from_node(last_index)
            # 补充终点
            path_coords.append([self.goal.x, self.goal.y, self.goal.z])
            last_time = time.time() - start_time
            print(f"[数据] 最终路径物理长度: {self.calculate_path_length(path_coords):.4f} 米")
            print(f"[数据] 最终路径计算时间: {last_time:.4f} 秒")
            return path_coords
    
        return None

# --------------------------sample-------------------------- 
    def sample_free(self):
        """
        随机采样一个点
        :return: 随机点的坐标 [x, y, z]
        """
        rnd_gen = random.Random()
        rnd = [rnd_gen.uniform(self.min_rand[0], self.max_rand[0]),
               rnd_gen.uniform(self.min_rand[1], self.max_rand[1]),
               rnd_gen.uniform(self.min_rand[2], self.max_rand[2])]
        return rnd
    
    def sample_goal(self, goal_sample_rate):
        """
        根据给定的采样率决定是否采样目标点
        :param goal_sample_rate: 采样目标点的概率（0-100）
        :return: 采样点的坐标 [x, y, z]
        """
        rnd_gen2 = random.Random()    
        if rnd_gen2.randint(0, 99) > goal_sample_rate:
            return self.sample_free()
        else:
            return [self.goal.x, self.goal.y, self.goal.z]
    
    def informed_sample(self):
        # 非 informed 时回默认
        if (not self.use_informed_sampling) or self.c_best == float("inf"):
            return [
                random.uniform(self.min_rand[0], self.max_rand[0]),
                random.uniform(self.min_rand[1], self.max_rand[1]),
                random.uniform(self.min_rand[2], self.max_rand[2]),
            ]

        # ------------------------------------------------------------------
        # 1. 计算椭球轴长
        # ------------------------------------------------------------------
        c_best = self.c_best
        c_min = self.c_min

        delta = max(c_best**2 - c_min**2, 0.0)
        a1 = c_best / 2.0
        a2 = math.sqrt(delta) / 2.0
        a3 = self.max_rand[2]  # Z 方向不受限制，或者通常取 a2

        L = np.diag([a1, a2, a3])

        # ------------------------------------------------------------------
        # 2. 采样单位球（均匀）
        # ------------------------------------------------------------------
        # 高斯方向 + 半径随机
        p = np.random.normal(0, 1, 3)
        p /= np.linalg.norm(p)
        r = np.random.random() ** (1/3)
        p = p * r

        # ------------------------------------------------------------------
        # 3. 计算旋转矩阵：将 x 轴对齐到 (goal - start)
        # ------------------------------------------------------------------
        start = np.array([self.start.x, self.start.y, self.start.z])
        goal = np.array([self.goal.x, self.goal.y, self.goal.z])

        direction = goal - start
        direction /= np.linalg.norm(direction)

        ex = np.array([1.0, 0.0, 0.0])
        v = np.cross(ex, direction)
        c_theta = np.dot(ex, direction)
        s_theta = np.linalg.norm(v)

        if s_theta < 1e-9:
            if c_theta > 0:
                R = np.eye(3)
            else:
                # 180° 旋转
                R = np.array([
                    [-1, 0, 0],
                    [ 0,-1, 0],
                    [ 0, 0, 1],
                ])
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + vx @ vx * ((1 - c_theta) / (s_theta**2))

        # ------------------------------------------------------------------
        # 4. 恢复全局坐标
        # ------------------------------------------------------------------
        center = (start + goal) / 2.0
        sample = R @ (L @ p) + center

        # ==================================================================
        # 5. [新增] 边界保护逻辑
        # ==================================================================
        # 检查生成点是否在地图范围内
        if (self.min_rand[0] <= sample[0] <= self.max_rand[0] and
            self.min_rand[1] <= sample[1] <= self.max_rand[1] and
            self.min_rand[2] <= sample[2] <= self.max_rand[2]):
            
            return sample.tolist()
        else:
            # 如果超出范围，回退到全局均匀随机采样
            return [
                random.uniform(self.min_rand[0], self.max_rand[0]),
                random.uniform(self.min_rand[1], self.max_rand[1]),
                random.uniform(self.min_rand[2], self.max_rand[2]),
            ]
    
#---------------------------steer--------------------------   
    def steer(self, from_node, to_node):
        """
        从 from_node 向 to_node 扩展一个新节点，新节点的成本和父节点都会确定
        :param from_node: 起始节点
        :param to_node: 目标节点
        :return: 新节点
        """
        dist = self.calc_distance(from_node, to_node)
        if dist <= self.expand_dis:
            new_node = Node(to_node.x, to_node.y, to_node.z)
            new_node.cost = from_node.cost + dist + self.risk_cost(new_node)
            new_node.parent = from_node
            return new_node
        else:
            theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            phi = math.atan2(to_node.z -from_node.z, math.sqrt((to_node.x - from_node.x)**2 + (to_node.y - from_node.y)**2))
            new_node = Node(from_node.x + self.expand_dis * math.cos(theta) * math.cos(phi),
                            from_node.y + self.expand_dis * math.sin(theta) * math.cos(phi),
                            from_node.z + self.expand_dis * math.sin(phi))
            new_node.parent = from_node
            new_node.cost = from_node.cost + self.expand_dis + self.risk_cost(new_node)
            
            return new_node
        
    def apf_steer(self, from_node, to_node):
        """
        从 from_node 向 to_node 扩展一个新节点，会定义cost和parent，使用 APF 方法确定方向
        :param from_node: 起始节点
        :param to_node: 目标节点
        :return: 新节点
        """
        # 计算原始方向向量
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # 计算吸引力（指向目标）
        if dist > 1e-3:
            att_direction = np.array([dx, dy, dz]) / dist
        else:
            att_direction = np.array([0, 0, 0])
        
        # 计算排斥力（远离障碍物）
        rep_force = np.array([0.0, 0.0, 0.0])
        
        for obstacle in self.obstacle_list:
            obs_x, obs_y, zmin, zmax, r_crash, r_risk = obstacle
            
            # 跳过不在 z 范围内的障碍物
            if not (zmin <= from_node.z <= zmax):
                continue
            
            # 计算节点到障碍物的水平距离
            obs_dx = from_node.x - obs_x
            obs_dy = from_node.y - obs_y
            dist_to_obs = math.sqrt(obs_dx**2 + obs_dy**2)
            
            # 检查是否在障碍物的风险半径内
            if dist_to_obs < r_risk:
                obs_direction = np.array([obs_dx, obs_dy, 0]) / dist_to_obs
            else:
                obs_direction = np.array([0, 0, 0])
                    
            rep_strength = 20 * (1.0 - dist_to_obs / r_risk) ** 2
            rep_force += rep_strength * obs_direction
        
        # 计算合力方向
        total_force = att_direction + rep_force
        
        # 归一化合力方向
        force_magnitude = np.linalg.norm(total_force)
        
        if force_magnitude > 1e-3:
            final_direction = total_force / force_magnitude
        else:
            # 合力很小时，使用随机方向
            random_dir = np.random.randn(3)
            final_direction = random_dir / np.linalg.norm(random_dir)
        
        # 创建新节点
        new_node = Node(
            from_node.x + self.expand_dis * final_direction[0],
            from_node.y + self.expand_dis * final_direction[1],
            from_node.z + self.expand_dis * final_direction[2]
        )
        
        # 计算新节点的代价
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.expand_dis + self.risk_cost(new_node)
        
        return new_node

# ----------------------------rewire----------------------------
    # def choose_parent(self, new_node, near_inds):
    #     """
    #     选择最佳父节点
    #     :param new_node: 新节点
    #     :param near_inds: 附近节点的索引
    #     :return: 更新后的新节点
    #     """
    #     if not near_inds:
    #         return None

    #     costs = []
    #     for i in near_inds:
    #         near_node = self.node_list[i]
    #         t_node = self.steer(near_node, new_node)
    #         if t_node and not self.check_collision(t_node) and not self.check_edge_collision(near_node, t_node):
    #             costs.append(near_node.cost + self.calc_distance(near_node, t_node) + self.risk_cost(t_node))
    #         else:
    #             costs.append(float("inf"))  # 碰撞或无法连接

    #     min_cost = min(costs)
    #     if min_cost == float("inf"):
    #         return None

    #     min_ind = near_inds[costs.index(min_cost)]
    #     # new node变化了
    #     new_node = self.steer(self.node_list[min_ind], new_node)
    #     new_node.cost = min_cost
    #     new_node.parent = self.node_list[min_ind]
    #     return new_node

    def choose_parent(self, new_node, near_inds):
        """
        选择最佳父节点，不steer直接连接
        :param new_node: 新节点
        :param near_inds: 附近节点的索引
        :return: 更新后的新节点
        
        """
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            if  not self.check_edge_collision(near_node, new_node):
                costs.append(near_node.cost + self.calc_distance(near_node, new_node) + self.risk_cost(new_node))
            else:
                costs.append(float("inf"))  # 碰撞或无法连接

        min_cost = min(costs)
        if min_cost == float("inf"):
            return None

        min_ind = near_inds[costs.index(min_cost)]
        # new node不变化，直接连接
        new_node.cost = min_cost
        new_node.parent = self.node_list[min_ind]
        return new_node     

    def rewire(self, new_node, near_inds):
        

        for i in near_inds:
            near_node = self.node_list[i]
            
            
            # 不生成新点，而是计算“如果直连”的距离
            dist_to_edge = self.calc_distance(new_node, near_node)

            # 计算新路径的 Cost
            new_cost = new_node.cost + dist_to_edge + self.risk_cost(near_node)

            # 只有 Cost 真的变小了，才做昂贵的碰撞检测
            if new_cost < near_node.cost:
                if not self.check_edge_collision(new_node, near_node):
                    # 只更新关系，不动坐标！
                    #print('check_rewire')

                    near_node.parent = new_node
                    near_node.cost = new_cost
                    self.propagate_cost_to_leaves(near_node)

    def propagate_cost_to_leaves(self, parent_node):
        '''
        递归更新子节点的成本
        '''
        #print('check_propagate')
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_distance(parent_node, node) + parent_node.cost + self.risk_cost(node)
                self.propagate_cost_to_leaves(node)
        
        return

   
        
    def get_nearest_node_index(self, node_list, rnd):
        """
        找到树中距离随机点最近的节点的索引
        :param node_list: 当前树中的节点列表
        :param rnd: 随机采样点的坐标 [x, y]
        :return: 最近节点的索引
        """
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 + (node.z - rnd[2]) ** 2 for node in node_list]
        return dlist.index(min(dlist))

    def search_best_goal_node(self):
        """
        在树中搜索距离目标点最近且可达的节点
        :return: 最佳目标节点的索引，如果不存在则返回 None
        """
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y, n.z) for n in self.node_list]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal)
            if not self.check_collision(t_node) and not self.check_edge_collision(t_node, self.goal):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None


    def generate_final_path_from_node(self, end_node_index):
        end_node = self.node_list[end_node_index]
        path = [[end_node.x, end_node.y, end_node.z]]
        node = end_node
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y, node.z])
        return path[::-1]

    def find_near_nodes(self, new_node):
        """
        找到新节点附近的节点索引
        :param new_node: 新节点
        :return: 附近节点的索引列表
        """
        nnode = len(self.node_list) + 1
        r = self.search_radius * math.pow(math.log(nnode) / nnode, 1.0/3.0)  # 动态调整搜索半径
        r = min(r, self.search_radius) # 限制最大搜索半径
        r = max(r, 1.5*self.expand_dis) # 限制最小搜索半径，这个如果删除了大部分节点都不会进入重连阶段，没有优化
        dlist = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 + (node.z - new_node.z) ** 2 for node in self.node_list]
        near_inds = [i for i in range(len(dlist)) if dlist[i] <= r ** 2]
        return near_inds
    
    # --------------------------工具函数--------------------------
    def check_collision(self, node):
        """
        判断节点是否进入 crash 区（绝对禁止）。
        使用距离 <= (agent_R_crash + obs_R_crash) 的标准（XY 平面内，且 Z 重叠）。
        """
        for cx, cy, z_min, z_max, obs_R_crash, obs_R_risk in self.obstacle_list:
            if z_min - self.R_crash <= node.z <= z_max + self.R_crash:
                dist_xy = math.hypot(node.x - cx, node.y - cy)
                if dist_xy <= (self.R_crash + obs_R_crash):
                    return True
        return False

    # def check_collision(self, node, near_obstacle_list=None):
    #     """
    #     判断节点是否进入 crash 区（绝对禁止）。
    #     支持传入局部障碍物列表 near_obstacle_list 以加速边检测。
    #     """
    #     # 1. 如果没有传入局部障碍物列表，则默认遍历全局障碍物
    #     if near_obstacle_list is None:
    #         near_obstacle_list = self.obstacle_list

    #     # 2. 遍历检查
    #     for cx, cy, z_min, z_max, obs_R_crash, obs_R_risk in near_obstacle_list:
    #         # 考虑机体半径，扩充 Z 轴上下界
    #         if (z_min - self.R_crash) <= node.z <= (z_max + self.R_crash):
    #             dx = node.x - cx
    #             dy = node.y - cy
    #             crash_threshold = self.R_crash + obs_R_crash
                
    #             # 采用平方和比较，剔除极度耗时的 math.sqrt 开方运算
    #             if (dx * dx + dy * dy) <= (crash_threshold * crash_threshold):
    #                 return True
                    
    #     return False
    
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
                    penalty += 0.0 / margin  # 权重可调，暂时调成0
        return penalty


    def calc_dist_to_goal(self, x, y, z):
        """
        计算给定点到目标点的欧几里得距离
        :param x: 点的 x 坐标
        :param y: 点的 y 坐标
        :param z: 点的 z 坐标
        :return: 到目标点的距离
        """
        return math.sqrt((x - self.goal.x) ** 2 + (y - self.goal.y) ** 2 + (z - self.goal.z) ** 2)
    
    # def check_edge_collision(self, n1, n2):
    #     """
    #     检查线段 n1->n2 是否与任何障碍的 crash 区相交（XY 投影 + Z 重叠）
    #     使用线段到圆心的最短距离与 crash 半径和比较。
    #     """
    #     for cx, cy, z_min, z_max, obs_R_crash, obs_R_risk in self.obstacle_list:
    #         # 如果 z 方向不重叠则跳过
    #         seg_z_min = min(n1.z, n2.z)
    #         seg_z_max = max(n1.z, n2.z)
    #         if seg_z_max < z_min or seg_z_min > z_max:
    #             continue

    #         # 计算线段到障碍中心 (cx,cy) 的最短距离（XY 平面）
    #         dist_xy = self.point_to_line_distance_xy(cx, cy, n1.x, n1.y, n2.x, n2.y)

    #         # 若最短距离小于等于 crash 判定阈值（agent + obs），视为碰撞
    #         if dist_xy <= (self.R_crash + obs_R_crash):
    #             return True
    #     return False
    def check_edge_collision(self, n1, n2, step_size=0.3):
        """
        检查线段 n1->n2 是否碰撞。
        策略：按固定步长离散化插值 -> 逐点调用全局 self.check_collision
        """
        dist = self.calc_distance(n1, n2)
        if dist == 0: 
            return False

        # 如果未指定步长，给一个默认的安全步长 (例如 1)
        if step_size is None:
            step_size = 1.0 

        # =========================================================
        # 【离散步进检测】直接对整条线段插值
        # =========================================================
        n_steps = int(dist / step_size) + 1
        dx = (n2.x - n1.x) / dist
        dy = (n2.y - n1.y) / dist
        dz = (n2.z - n1.z) / dist

        for i in range(n_steps):
            cx = n1.x + dx * step_size * i
            cy = n1.y + dy * step_size * i
            cz = n1.z + dz * step_size * i
            
            # 直接调用单点检测（不传 near_obstacle_list，默认使用全局全量检测）
            if self.check_collision(Node(cx, cy, cz)):
                return True
        
        # 别忘了检查最后一个终点
        if self.check_collision(n2):
            return True
            
        return False
    
    def point_to_line_distance_xy(self, px, py, x1, y1, x2, y2):
        """计算点到线段的最短距离（仅在 XY 平面）"""
        # 向量 AP
        apx = px - x1
        apy = py - y1

        # 向量 AB
        abx = x2 - x1
        aby = y2 - y1

        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq == 0:
            return math.sqrt(apx * apx + apy * apy)

        t = max(0, min(1, (apx * abx + apy * aby) / ab_len_sq))

        closest_x = x1 + t * abx
        closest_y = y1 + t * aby

        dx = px - closest_x
        dy = py - closest_y
        return math.sqrt(dx * dx + dy * dy)

    def point_to_line_distance(self, px, py, pz, x1, y1, z1, x2, y2, z2):
        """计算点到线段的距离"""
        # 向量AP
        apx = px - x1
        apy = py - y1
        apz = pz - z1
        # 向量AB
        abx = x2 - x1
        aby = y2 - y1
        abz = z2 - z1
        
        # 计算投影长度
        dot = apx * abx + apy * aby + apz * abz
        ab_len_sq = abx * abx + aby * aby + abz * abz
        
        if ab_len_sq == 0:
            return math.sqrt(apx * apx + apy * apy + apz * apz)
        
        t = max(0, min(1, dot / ab_len_sq))
        
        # 最近点坐标
        closest_x = x1 + t * abx
        closest_y = y1 + t * aby
        closest_z = z1 + t * abz
        
        # 计算距离
        dx = px - closest_x
        dy = py - closest_y
        dz = pz - closest_z
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    def calc_distance(self, node1, node2):
        """
        计算两个节点之间的欧几里得距离
        :param node1: 第一个节点
        :param node2: 第二个节点
        :return: 距离
        """
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 + (node1.z - node2.z) ** 2)
    

    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            length += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
        return length

def calculate_path_length(path):
    length = 0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        length += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    return length


def extract_waypoints(points_norm, scores, map_dim, eps=0.02, peak_radius=0.02):
    if len(points_norm) == 0: 
        return np.empty((0, 3))
        
    Lx, Ly, Lz = map_dim
    center = 0.5 * np.array([Lx, Ly, Lz])
    scale = max(Lx, Ly, Lz)
    points_phys = points_norm * scale + center
    dim_scale = np.array([Lx, Ly, Lz])
    points_ratio = points_phys / dim_scale
    
    # 1. 构建 KDTree
    tree = KDTree(points_ratio)
    
    # 2. 批量查询：一次性算出所有点在半径内的邻居索引列表 (底层 C 语言多线程执行)
    neighbors_list = tree.query_ball_point(points_ratio, r=peak_radius)
    
    # 3. 向量化判断局部极大值
    is_peak = np.ones(len(points_ratio), dtype=bool) 
    for i, neighbors in enumerate(neighbors_list):
        if np.any(scores[neighbors] > scores[i]):
            is_peak[i] = False
            
    peaks_ratio = points_ratio[is_peak]
    peaks_norm = points_norm[is_peak]
    peak_scores = scores[is_peak]

    if len(peaks_ratio) == 0: 
        return np.empty((0, 3))
        
    # 4. DBSCAN 聚类 (点数已被极度压缩，瞬间完成)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(peaks_ratio)
    labels = clustering.labels_
    
    waypoints_norm = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1: continue
        mask = (labels == label)
        cluster_norm = peaks_norm[mask]
        cluster_scores = peak_scores[mask]
        best_idx = np.argmax(cluster_scores)
        waypoints_norm.append(cluster_norm[best_idx])
        
    return np.array(waypoints_norm)

# ==========================================
# 2. 辅助函数 (从 save_sample3 中提取)
# ==========================================
def check_collision(point, obstacles):
    px, py, pz = point
    for (ox, oy, zmin, zmax, r_crash, _) in obstacles:
        if zmin <= pz <= zmax:
            if (px - ox)**2 + (py - oy)**2 <= r_crash**2:
                return True
    return False

def get_min_distance_to_obstacles(point, obstacles):
    px, py, pz = point
    min_dist = float('inf')
    for (ox, oy, zmin, zmax, r_crash, _) in obstacles:
        d_hor = np.sqrt((px - ox)**2 + (py - oy)**2) - r_crash
        if pz > zmax: d_ver = pz - zmax
        elif pz < zmin: d_ver = zmin - pz
        else: d_ver = 0.0

        if d_hor > 0 and d_ver == 0: dist = d_hor
        elif d_hor <= 0 and d_ver > 0: dist = d_ver
        elif d_hor > 0 and d_ver > 0: dist = np.sqrt(d_hor**2 + d_ver**2)
        else: dist = 0.0 
        
        if dist < min_dist: min_dist = dist
    return min_dist

def get_nearest_obstacle_normal(point, obstacles, eps=1e-8):
    px, py, pz = point
    min_dist = float('inf')
    best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for (ox, oy, zmin, zmax, r_crash, _) in obstacles:
        dx = px - ox
        dy = py - oy
        d_xy = np.sqrt(dx**2 + dy**2) + eps 
        
        ux, uy = dx / d_xy, dy / d_xy
        dist_xy = d_xy - r_crash
        
        dist_z, vz = 0.0, 0.0
        if pz > zmax:
            dist_z, vz = pz - zmax, 1.0
        elif pz < zmin:
            dist_z, vz = zmin - pz, -1.0

        if dist_xy > 0 and dist_z <= 0:   curr_dist = dist_xy
        elif dist_xy <= 0 and dist_z > 0: curr_dist = dist_z
        elif dist_xy > 0 and dist_z > 0:  curr_dist = np.sqrt(dist_xy**2 + dist_z**2)
        else: curr_dist = 0.0
        
        if curr_dist < min_dist:
            min_dist = curr_dist
            nx, ny, nz = 0.0, 0.0, 0.0
            
            if dist_xy > 0 and dist_z <= 0:   nx, ny, nz = ux, uy, 0.0
            elif dist_xy <= 0 and dist_z > 0: nx, ny, nz = 0.0, 0.0, vz
            elif dist_xy > 0 and dist_z > 0:  
                vx, vy, vz_vec = dist_xy * ux, dist_xy * uy, dist_z * vz
                norm = np.sqrt(vx**2 + vy**2 + vz_vec**2) + eps
                nx, ny, nz = vx/norm, vy/norm, vz_vec/norm
            else:
                nx, ny, nz = ux, uy, 0.0
                
            best_normal = np.array([nx, ny, nz], dtype=np.float32)
    return best_normal

def check_point_validity(x, y, z, obstacles):
    """适配器：利用现有的 check_collision 检查合法性"""
    return not check_collision([x, y, z], obstacles)

def generate_valid_tasks(num_tasks, env_map, min_dist=200.0, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    tasks = []
    obs_list = env_map["obstacles"]
    x_size, y_size, z_size = env_map["map_dim"]
    
    while len(tasks) < num_tasks:
        sx = random.uniform(0, x_size)
        sy = random.uniform(0, y_size)
        sz = random.uniform(0, z_size)
        
        gx = random.uniform(0, x_size)
        gy = random.uniform(0, y_size)
        gz = random.uniform(0, z_size)

        if not check_point_validity(sx, sy, sz, obs_list): continue
        if not check_point_validity(gx, gy, gz, obs_list): continue

        dist = math.sqrt((sx - gx)**2 + (sy - gy)**2 + (sz - gz)**2)
        if dist < min_dist: continue

        tasks.append(([sx, sy, sz], [gx, gy, gz]))
    return tasks


def filter_zigzag_waypoints(start_pt, goal_pt, mid_wps, min_dist=0.03, local_thresh=0.15, max_turn_angle=60.0):
    if len(mid_wps) == 0: 
        return np.empty((0, 3)) # ⭐ 防爆保底1

    d_s, d_g = np.linalg.norm(mid_wps - start_pt, axis=1), np.linalg.norm(mid_wps - goal_pt, axis=1)
    ordered_wps = mid_wps[np.argsort(d_s / (d_s + d_g + 1e-6))]
    
    path = [start_pt] + list(ordered_wps) + [goal_pt]
    i = 1
    while i < len(path) - 1:
        v_in, v_out = path[i] - path[i-1], path[i+1] - path[i]
        n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if n_in < min_dist or n_out < min_dist:
            path.pop(i); continue
        if n_in > 1e-6 and n_out > 1e-6 and (n_in < local_thresh or n_out < local_thresh):
            if np.degrees(np.arccos(np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0))) > max_turn_angle:
                path.pop(i); continue
        i += 1
        
    # ⭐ 防爆保底2：如果中间节点被全部剔除，强制返回 (0,3) 的二维空数组
    filtered_wps = np.array(path[1:-1])
    if len(filtered_wps) == 0:
        return np.empty((0, 3))
        
    return filtered_wps

def generate_point_cloud_features(S, G, map_dim, obstacles, N_attempts=4096):
    """
    极速向量化版的点云采样与 9 维特征构造函数
    """
    Lx, Ly, Lz = map_dim
    xyz_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    xyz_max = np.array([Lx, Ly, Lz], dtype=np.float32)
    center = 0.5 * (xyz_min + xyz_max)
    scale = max(Lx, Ly, Lz)
    eps = 1e-8

    # 随机撒点
    candidates = np.random.uniform(xyz_min, xyz_max, size=(N_attempts, 3))
    
    # 提取障碍物矩阵
    obs_arr = np.array(obstacles, dtype=np.float32)
    ox, oy, zmin, zmax, r_crash = obs_arr[:, 0], obs_arr[:, 1], obs_arr[:, 2], obs_arr[:, 3], obs_arr[:, 4]

    # --- 1. 向量化碰撞检测 ---
    c_x, c_y, c_z = candidates[:, 0:1], candidates[:, 1:2], candidates[:, 2:3]
    in_z = (c_z >= zmin[None, :]) & (c_z <= zmax[None, :])
    in_xy = ((c_x - ox[None, :])**2 + (c_y - oy[None, :])**2) <= (r_crash[None, :]**2)
    valid_pts = candidates[~np.any(in_z & in_xy, axis=1)]
    
    # 拼接有效点云
    xyz = np.vstack([S[None], G[None], valid_pts]) 
    V = xyz.shape[0]
    
    # --- 2. 向量化计算最近距离 ---
    p_x, p_y, p_z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    dx, dy = p_x - ox[None, :], p_y - oy[None, :]
    dist_xy = np.sqrt(dx**2 + dy**2) + eps - r_crash[None, :]
    
    dist_z = np.zeros((V, len(obstacles)), dtype=np.float32)
    vz = np.zeros((V, len(obstacles)), dtype=np.float32)
    above, below = p_z > zmax[None, :], p_z < zmin[None, :]
    dist_z[above], vz[above] = (p_z - zmax[None, :])[above], 1.0
    dist_z[below], vz[below] = (zmin[None, :] - p_z)[below], -1.0
    
    curr_dist = np.zeros((V, len(obstacles)), dtype=np.float32)
    c1, c2, c3 = (dist_xy > 0) & (dist_z <= 0), (dist_xy <= 0) & (dist_z > 0), (dist_xy > 0) & (dist_z > 0)
    curr_dist[c1], curr_dist[c2], curr_dist[c3] = dist_xy[c1], dist_z[c2], np.sqrt(dist_xy[c3]**2 + dist_z[c3]**2)
    
    d_obs = np.min(curr_dist, axis=1)
    nearest_idx = np.argmin(curr_dist, axis=1)
    
    # --- 3. 仅对最近障碍物计算法向量 ---
    n_ox, n_oy, n_zmin, n_zmax, n_r_crash = ox[nearest_idx], oy[nearest_idx], zmin[nearest_idx], zmax[nearest_idx], r_crash[nearest_idx]
    n_dx, n_dy, n_pz = xyz[:, 0] - n_ox, xyz[:, 1] - n_oy, xyz[:, 2]
    n_d_xy = np.sqrt(n_dx**2 + n_dy**2) + eps
    ux, uy, n_dist_xy = n_dx / n_d_xy, n_dy / n_d_xy, n_d_xy - n_r_crash
    
    n_dist_z, n_vz = np.zeros(V, dtype=np.float32), np.zeros(V, dtype=np.float32)
    n_above, n_below = n_pz > n_zmax, n_pz < n_zmin
    n_dist_z[n_above], n_vz[n_above] = n_pz[n_above] - n_zmax[n_above], 1.0
    n_dist_z[n_below], n_vz[n_below] = n_zmin[n_below] - n_pz[n_below], -1.0
    
    nx, ny, nz = np.zeros(V, dtype=np.float32), np.zeros(V, dtype=np.float32), np.zeros(V, dtype=np.float32)
    c1, c2, c3 = (n_dist_xy > 0) & (n_dist_z <= 0), (n_dist_xy <= 0) & (n_dist_z > 0), (n_dist_xy > 0) & (n_dist_z > 0)
    nx[c1], ny[c1], nz[c2] = ux[c1], uy[c1], n_vz[c2]
    
    v_x, v_y, v_z = n_dist_xy[c3] * ux[c3], n_dist_xy[c3] * uy[c3], n_dist_z[c3] * n_vz[c3]
    norm_vec = np.sqrt(v_x**2 + v_y**2 + v_z**2) + eps
    nx[c3], ny[c3], nz[c3] = v_x / norm_vec, v_y / norm_vec, v_z / norm_vec
    
    c_else = ~(c1 | c2 | c3)
    nx[c_else], ny[c_else], nz[c_else] = ux[c_else], uy[c_else], 0.0
    obs_normals = np.stack([nx, ny, nz], axis=1)
    
    # --- 4. 组装 9 维特征 ---
    xyz_norm = (xyz - center) / (scale + eps)
    d_obs_norm = d_obs / (scale + eps)
    d_s = np.linalg.norm(xyz - S[None], axis=1)
    d_g = np.linalg.norm(xyz - G[None], axis=1)
    
    features_9d = np.stack([
        xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2], 
        d_g / (d_s + d_g + eps), d_s / (d_s + d_g + eps), 
        d_obs_norm, 
        obs_normals[:, 0], obs_normals[:, 1], obs_normals[:, 2]
    ], axis=1).astype(np.float32)
    
    # 返回主程序所需的所有变量
    return features_9d, xyz, xyz_norm, center, scale

def visualize_prediction_results(Lx, Ly, Lz, obstacles, S, G, sorted_wps, final_waypoints, 
                                 xyz, high_score_xyz, high_score_vals, score_threshold):
    """
    专业的 3D 双子图可视化函数
    
    参数:
        Lx, Ly, Lz: 地图的长、宽、高
        obstacles: 障碍物列表
        S, G: 起点和终点坐标 (1D array)
        sorted_wps: 提取并排序后的中间航路点 (N x 3 array)
        final_waypoints: 包含起终点的完整航路点序列 (N x 3 array)
        xyz: 所有的采样点云集 (N x 3 array)
        high_score_xyz: 得分高于阈值的采样点坐标 (M x 3 array)
        high_score_vals: 高于阈值的具体得分 (M array)
        score_threshold: 过滤得分的阈值 (float)
    """
    print("\n正在启动 3D 可视化窗口...")
    
    scatter_kwargs = {
        'cmap': 'jet', 's': 20, 'vmin': 0.0, 'vmax': 1.0, 
        'alpha': 0.5, 'edgecolor': 'none'
    }

    fig = plt.figure(figsize=(16, 8))
    
    # 内部辅助函数：绘制障碍物
    def plot_obstacles(ax):
        obs_color = [0.5, 0.5, 0.5, 0.3] 
        for (ox, oy, zmin, zmax, r_crash, _) in obstacles:
            theta, z = np.linspace(0, 2 * np.pi, 20), np.linspace(zmin, zmax, 2)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid, y_grid = ox + r_crash * np.cos(theta_grid), oy + r_crash * np.sin(theta_grid)
            ax.plot_surface(x_grid, y_grid, z_grid, color=obs_color, edgecolor='none', shade=True)
            ax.plot_trisurf(x_grid.flatten(), y_grid.flatten(), zmin * np.ones_like(x_grid.flatten()), color=obs_color, edgecolor='none', shade=False)
            ax.plot_trisurf(x_grid.flatten(), y_grid.flatten(), zmax * np.ones_like(x_grid.flatten()), color=obs_color, edgecolor='none', shade=False)

    # 内部辅助函数：设置坐标轴属性
    def set_ax_properties(ax, title):
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_zlim(0, Lz)
        ax.set_box_aspect([Lx, Ly, Lz]) 
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=30, azim=-60)

    # --- 左图：Obstacles & Final Path ---
    ax1 = fig.add_subplot(121, projection='3d')
    plot_obstacles(ax1)
    
    ax1.plot(final_waypoints[:, 0], final_waypoints[:, 1], final_waypoints[:, 2], 
             color='black', linewidth=2, linestyle='--', label='Planned Path')
             
    ax1.scatter(*S, c='green', s=150, marker='s', edgecolor='black', label='Start', zorder=5)
    ax1.scatter(*G, c='blue', s=150, marker='s', edgecolor='black', label='Goal', zorder=5)
    
    if len(sorted_wps) > 0:
        ax1.scatter(sorted_wps[:, 0], sorted_wps[:, 1], sorted_wps[:, 2], 
                    c='magenta', s=150, marker='*', edgecolor='black', label='Extracted WPs', zorder=4)

    set_ax_properties(ax1, f'Obstacles & Final Path\n(Found {len(sorted_wps)} Mid Waypoints)')
    ax1.legend(loc='upper right')

    # --- 右图：Score Heatmap & Waypoints ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='gray', s=1, alpha=0.05)

    p2 = None
    if len(high_score_xyz) > 0:
        p2 = ax2.scatter(high_score_xyz[:, 0], high_score_xyz[:, 1], high_score_xyz[:, 2],
                         c=high_score_vals, label='Pred Heatmap', **scatter_kwargs)
                         
    ax2.scatter(*S, c='green', s=150, marker='s', edgecolor='black', label='Start', zorder=5)
    ax2.scatter(*G, c='blue', s=150, marker='s', edgecolor='black', label='Goal', zorder=5)
    
    if len(sorted_wps) > 0:
        ax2.scatter(sorted_wps[:, 0], sorted_wps[:, 1], sorted_wps[:, 2], 
                    c='magenta', s=150, marker='*', edgecolor='black', label='Extracted WPs', zorder=4)

    set_ax_properties(ax2, f'Prediction (Score > {score_threshold})\nHeatmap & Waypoints')
    ax2.legend(loc='upper right')

    # 统一颜色条
    plot_handle = p2
    if plot_handle is None:
        plot_handle = ax2.scatter([xyz[0,0]], [xyz[0,1]], [xyz[0,2]], c=[0.0], s=0, **scatter_kwargs)

    cbar = fig.colorbar(plot_handle, ax=[ax1, ax2], shrink=0.7, location='right', pad=0.02)
    cbar.set_label('Prediction Score')

    print("可视化窗口已打开，你可以用鼠标分别旋转、缩放两个子图。")
    plt.show()

def count_path_collisions(path_coords, obstacles, r_agent_crash=1.2):
    """
    精确统计路径线段穿过了多少个障碍物
    """
    if len(path_coords) < 2: 
        return 0
        
    total_collisions = 0
    for obs in obstacles:
        cx, cy, z_min, z_max, obs_r_crash, _ = obs
        hit = False
        
        # 检查这条路径的所有线段，是否有任何一段撞到了当前障碍物
        for i in range(len(path_coords) - 1):
            p1, p2 = path_coords[i], path_coords[i+1]
            
            # 1. Z 轴未重叠，安全
            if max(p1[2], p2[2]) < z_min or min(p1[2], p2[2]) > z_max:
                continue
            
            # 2. XY 平面到线段的最短距离
            apx, apy = cx - p1[0], cy - p1[1]
            abx, aby = p2[0] - p1[0], p2[1] - p1[1]
            ab_len_sq = abx**2 + aby**2
            
            if ab_len_sq == 0:
                dist_xy = math.hypot(apx, apy)
            else:
                t = max(0, min(1, (apx*abx + apy*aby) / ab_len_sq))
                dist_xy = math.hypot(cx - (p1[0] + t*abx), cy - (p1[1] + t*aby))
            
            # 3. 距离小于碰撞半径之和，发生碰撞
            if dist_xy <= (r_agent_crash + obs_r_crash):
                hit = True
                break  # 只要有一段撞了这个障碍物，就算穿过了该障碍物，跳出内层循环
                
        if hit:
            total_collisions += 1
            
    return total_collisions


def plot_uav_comparison(env_map, final_waypoints, final_path_vec, final_path_rrt):
    """
    可视化函数：展示三幅对比图
    修复了 plot_surface 导致的颜色广播错误
    """
    obstacles = env_map.get("obstacles", [])
    Lx, Ly, Lz = env_map["map_dim"]
    
    # 确保有起终点数据
    if final_waypoints is None or len(final_waypoints) < 2:
        print("[绘图警告] 无效的航路点数据")
        return
        
    S = final_waypoints[0]
    G = final_waypoints[-1]

    fig = plt.figure(figsize=(20, 7))
    
    plot_configs = [
        {"title": "1. Predicted Waypoints & Obstacles", "path": final_waypoints, "is_waypoint": True},
        {"title": "2. Vectorized RRT Path", "path": final_path_vec, "is_waypoint": False},
        {"title": "3. Standard RRT* Path", "path": final_path_rrt, "is_waypoint": False}
    ]

    for i, config in enumerate(plot_configs):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # --- 绘制障碍物 (圆柱体) ---
        if len(obstacles) > 0:
            for obs in obstacles:
                # 兼容不同长度的 obstacle 定义
                cx, cy, zmin, zmax, r_crash = obs[0], obs[1], obs[2], obs[3], obs[4]
                
                # 构建圆柱体网格
                z_grid = np.array([[zmin, zmin], [zmax, zmax]])
                theta = np.linspace(0, 2*np.pi, 20)
                x_circle = r_crash * np.cos(theta) + cx
                y_circle = r_crash * np.sin(theta) + cy
                
                # 将圆周坐标转为网格
                X = np.tile(x_circle, (2, 1))
                Y = np.tile(y_circle, (2, 1))
                Z = np.array([np.ones(20)*zmin, np.ones(20)*zmax])

                # 使用明确的 facecolor 避免 shade 导致的颜色数组空值错误
                ax.plot_surface(X, Y, Z, color='red', alpha=0.2, shade=False)

        # --- 绘制路径 ---
        path = config["path"]
        if path is not None:
            path_np = np.array(path)
            if len(path_np) > 0:
                if config["is_waypoint"]:
                    # 航路点用点划线
                    ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], 
                            color='green', linestyle='--', marker='o', 
                            markersize=3, linewidth=1, label='Net Waypoints')
                else:
                    # RRT 路径用实线
                    ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], 
                            color='blue', linestyle='-', linewidth=2, label='RRT Path')
        
        # --- 绘制任务起终点 ---
        ax.scatter(S[0], S[1], S[2], color='gold', s=120, marker='*', edgecolors='black', label='Start', zorder=5)
        ax.scatter(G[0], G[1], G[2], color='magenta', s=100, marker='X', edgecolors='black', label='Goal', zorder=5)

        # --- 坐标轴与比例设置 ---
        ax.set_title(config["title"], fontsize=12)
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_zlim(0, Lz)
        
        # 核心：设置真实比例，防止 z 轴被拉伸，Lx:Ly:Lz = 1500:1500:240
        ax.set_box_aspect((Lx, Ly, Lz)) 
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

    plt.tight_layout()
    plt.show()

# ==========================================
# 3. 主程序流水线
# ==========================================
if __name__ == '__main__':
    import time
    import random
    import math
    import numpy as np
    import torch
    
    # ==========================================
    # ⭐ 阶段 0: 核心工程加载与全局预热 (仅执行一次)
    # ==========================================
    print("\n[系统初始化] 正在加载模型与预热底层环境...")
    
    # 1. CPU 数学库预热
    dummy_pts = np.random.rand(10, 3)
    dummy_scores = np.random.rand(10)
    _ = extract_waypoints(dummy_pts, dummy_scores, map_dim=(100,100,100), eps=0.1, peak_radius=0.1)
    
    # 2. 加载模型到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=1, input_dim=9, dropout_p=0.0).to(device) 
    # model_path = r"C:\Users\Administrator\Desktop\experiments\best_model_for_trian_data5\best_model.pth"
    model_path = r"C:\Users\Administrator\Desktop\experiments\best_model_for_train_data6\best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. GPU 显存预热
    dummy_tensor = torch.rand(1, 9, 2048).to(device)
    with torch.no_grad(): _ = model(dummy_tensor)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    print("  --> 模型加载与底层引擎预热全部完成！\n")

    # ==========================================
    # ⭐ 阶段 1: 批量测试参数初始化
    # ==========================================
    num_tests = 5
    
    # 时间统计
    sum_time_step2_feat = 0.0
    sum_time_step3_infer = 0.0
    sum_time_step4_cluster = 0.0
    sum_time_rrt_star = 0.0
    sum_time_rrt_star_success = 0.0
    
    # ⭐ 新增：向量化 RRT 的时间统计
    sum_time_vector_rrt = 0.0
    sum_time_vector_rrt_success = 0.0
    
    # 长度与碰撞统计
    sum_len_net_wp = 0.0
    sum_len_rrt_star = 0.0
    sum_len_vector_rrt = 0.0  # ⭐ 新增：向量化 RRT 的长度统计
    sum_obs_collision_ratio = 0.0  
    
    # 成功次数统计
    valid_wp_count = 0
    success_rrt_count = 0
    success_vector_rrt_count = 0  # ⭐ 新增：向量化 RRT 的成功次数统计

    print(f"================ 开始执行 {num_tests} 组批量测试 ================")

    for i in range(num_tests):
        print(f"\n>>> 正在运行测试 [{i+1}/{num_tests}] ...", end=" ", flush=True)
        
        # ----------------------------------
        # 1. 随机生成环境与任务
        # ----------------------------------
        # env_map = env_generator(
        #     rho=random.uniform(0.6, 0.85), 
        #     map_dim=(1500, 1500, 240),
        #     r_crash_range=(30, 50),
        #     r_risk_range=(3, 7),
        #     zmax_range=(30, 240),
        #     max_iter=5000,
        #     seed=None
        # )

        env_map = env_generator_cluster(
                    map_dim=(1500, 1500, 240),   # (Lx, Ly, Lz)
                    num_clusters=20,             # 建议 10~15 之间，保证有足够空间
                    chain_length_range=(1, 4),   # 每个簇的圆柱体数量
                    r_center_range=(100, 150),    # 接近地图中心的圆柱体半径范围
                    r_edge_range=(20, 50),       # 接近地图边缘的圆柱体半径范围
                    r_risk_range=(10, 20),       # 风险半径偏移量
                    zmax_range=(240, 240),
                    min_center_dist=200,         # 【核心参数】任意两个簇中心点的最小绝对距离！
                    seed=None,
                )
        Lx, Ly, Lz = env_map["map_dim"]
        obstacles = env_map["obstacles"]
        scale = max(Lx, Ly, Lz)
        center = 0.5 * np.array([Lx, Ly, Lz])
        
        tasks = generate_valid_tasks(num_tasks=1, env_map=env_map, min_dist=1200, seed=None)
        S, G = np.array(tasks[0][0], dtype=np.float32), np.array(tasks[0][1], dtype=np.float32)

        # ----------------------------------
        # 2. 采样与特征构造
        # ----------------------------------
        t2_start = time.time()
        features_9d, xyz, xyz_norm, center, scale = generate_point_cloud_features(
            S=S, G=G, map_dim=(Lx, Ly, Lz), obstacles=obstacles, N_attempts=4096
        )
        t2_end = time.time()
        sum_time_step2_feat += (t2_end - t2_start)

        # ----------------------------------
        # 3. 纯网络前向推理
        # ----------------------------------
        input_tensor = torch.tensor(features_9d).transpose(0, 1).unsqueeze(0).to(device)
        t3_start = time.time()
        with torch.no_grad(): 
            pred_scores = model(input_tensor) 
        scores_np = pred_scores.squeeze().cpu().numpy()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t3_end = time.time()
        sum_time_step3_infer += (t3_end - t3_start)

        # ----------------------------------
        # 4. 极速聚类与后处理
        # ----------------------------------
        t4_start = time.time()
        score_threshold = 0.8 
        mask = scores_np > score_threshold
        
        high_score_norm, high_score_vals = xyz_norm[mask], scores_np[mask]
        
        if len(high_score_norm) > 0:
            extracted_wps_norm = extract_waypoints(
                points_norm=high_score_norm, scores=high_score_vals, 
                map_dim=(Lx, Ly, Lz), 
                eps=0.15,           
                peak_radius=0.15    
            )
            if len(extracted_wps_norm) > 0:
                extracted_wps_physical = extracted_wps_norm * scale + center
                sorted_wps = filter_zigzag_waypoints(
                    start_pt=S, goal_pt=G, mid_wps=extracted_wps_physical,
                    min_dist=0.05 * scale, local_thresh=0.2 * scale, max_turn_angle=60.0
                )
            else: sorted_wps = np.empty((0, 3))
        else: sorted_wps = np.empty((0, 3))

        final_waypoints = np.vstack([S[None], sorted_wps, G[None]])
        t4_end = time.time()
        sum_time_step4_cluster += (t4_end - t4_start)

        print("") # 换行输出日志
        
        # --- 记录网络提取航路点的长度与碰撞比例 ---
        if len(final_waypoints) > 1:
            wp_path_length = np.sum(np.linalg.norm(np.diff(final_waypoints, axis=0), axis=1))
            sum_len_net_wp += wp_path_length
            
            num_obs = len(obstacles)
            collisions = count_path_collisions(final_waypoints, obstacles, r_agent_crash=1.2)
            col_ratio = collisions / num_obs if num_obs > 0 else 0.0
            sum_obs_collision_ratio += col_ratio
            
            valid_wp_count += 1
            print(f"  ├─ 网络生成航路点数量: {len(final_waypoints)}, 总长: {wp_path_length:.2f}m, 穿障率: {col_ratio*100:.1f}% ({collisions}/{num_obs})")
        else:
            print("  ├─ 网络未能生成有效航路点")


        # ----------------------------------
        # 4.5 向量化 RRT 规划
        # ----------------------------------
        planner_vector = RRT_vector(
            waypoints=final_waypoints, R_crash=1.2, R_risk=1.7, 
            obstacle_list=obstacles, expand_dis=15, max_iter=3000, search_until_max_iter=False
        )
        
        t_vec_start = time.time() # ⭐ 新增：开始计时
        found_status, time_list, iter_list, length_list, raw_segments, final_path = planner_vector.planning()
        t_vec_end = time.time()   # ⭐ 新增：结束计时
        
        vec_time_cost = t_vec_end - t_vec_start
        sum_time_vector_rrt += vec_time_cost # ⭐ 累加整体耗时

        if all(found_status) and final_path is not None:
            # ⭐ 统计成功数据
            success_vector_rrt_count += 1
            sum_time_vector_rrt_success += vec_time_cost
            total_vec_len = sum(length_list)
            sum_len_vector_rrt += total_vec_len
            
            print(f"  ├─ 向量化RRT规划成功! 总耗时: {vec_time_cost:.3f}s, 总长度: {total_vec_len:.2f}m")
            # print("\n🎉 规划大获全胜！所有航段均已顺利连通！")
            # print(f"📊 数据统计:")
            # for i in range(len(found_status)):
            #     print(f"  - 航段 {i+1}: 耗时 {time_list[i]:.4f}秒, 迭代 {iter_list[i]}次, 长度 {length_list[i]:.2f}米")
        else:
            failed_segs = [i for i, found in enumerate(found_status) if not found]
            print(f"  ├─ 向量化RRT规划失败! 耗时: {vec_time_cost:.3f}s, 未连通航段: {failed_segs}")


        # ----------------------------------
        # 5. 标准 RRT* 规划对比测试
        # ----------------------------------
        planner = RRTStar(
            start=final_waypoints[0], goal=final_waypoints[-1], 
            R_crash=1.2, R_risk=1.7, 
            obstacle_list=obstacles, 
            rand_area=[[0, 0, 0], [Lx, Ly, Lz]],
            expand_dis=15, max_iter=10000, search_radius=75,
            search_until_max_iter=False
        )
        
        trrt_start = time.time()
        rrt_path = planner.planning()
        trrt_end = time.time()
        
        sum_time_rrt_star += (trrt_end - trrt_start)
        
        if rrt_path is not None:
            plen = planner.calculate_path_length(rrt_path)
            sum_len_rrt_star += plen
            sum_time_rrt_star_success += (trrt_end - trrt_start) 
            success_rrt_count += 1
            print(f"  └─ 标准RRT*规划成功! 耗时: {trrt_end - trrt_start:.3f}s, 长度: {plen:.2f}m")
        else:
            print(f"  └─ 标准RRT*规划失败! 耗时: {trrt_end - trrt_start:.3f}s")
    
        if i <= 2:
            plot_uav_comparison(
                env_map=env_map, 
                final_waypoints=final_waypoints, 
                final_path_vec=final_path,   # 向量化RRT的结果
                final_path_rrt=rrt_path      # 标准RRT*的结果
            )
            

    # ==========================================
    # ⭐ 阶段 2: 统计并输出平均指标
    # ==========================================
    print("\n" + "=" * 60)
    print(f"🏆 {num_tests} 组批量测试完成！最终统计报告：")
    print("=" * 60)
    
    # 1. 平均时间计算
    avg_rrt_success_time = (sum_time_rrt_star_success / success_rrt_count) if success_rrt_count > 0 else 0
    avg_vec_success_time = (sum_time_vector_rrt_success / success_vector_rrt_count) if success_vector_rrt_count > 0 else 0 # ⭐ 新增

    print("[⏱️ 平均耗时统计]")
    print(f"  1. 采样与特征构造 (Step 2) : {sum_time_step2_feat / num_tests:.4f} 秒")
    print(f"  2. 模型纯前向推理 (Step 3) : {sum_time_step3_infer / num_tests:.4f} 秒")
    print(f"  3. 极速聚类与过滤 (Step 4) : {sum_time_step4_cluster / num_tests:.4f} 秒")
    print(f"  4. 向量化 RRT 整体平均耗时 : {sum_time_vector_rrt / num_tests:.4f} 秒 (含失败兜底)")    # ⭐ 新增
    print(f"  5. 向量化 RRT 成功求解耗时 : {avg_vec_success_time:.4f} 秒 (仅算成功的 {success_vector_rrt_count} 次)") # ⭐ 新增
    print(f"  6. 标准 RRT* 整体平均耗时  : {sum_time_rrt_star / num_tests:.4f} 秒 (含失败兜底)")
    print(f"  7. 标准 RRT* 成功求解耗时  : {avg_rrt_success_time:.4f} 秒 (仅算成功的 {success_rrt_count} 次)")
    
    print("-" * 60)
    
    # 2. 平均长度计算
    avg_len_wp = (sum_len_net_wp / valid_wp_count) if valid_wp_count > 0 else 0
    avg_obs_col_ratio = (sum_obs_collision_ratio / valid_wp_count) * 100 if valid_wp_count > 0 else 0
    avg_len_rrt = (sum_len_rrt_star / success_rrt_count) if success_rrt_count > 0 else 0
    avg_len_vec = (sum_len_vector_rrt / success_vector_rrt_count) if success_vector_rrt_count > 0 else 0 # ⭐ 新增
    
    print("[📏 平均路径指标统计]")
    print(f"  1. 航路点连线平均穿障率      : {avg_obs_col_ratio:.2f}% (基于有效网络输出)")
    print(f"  2. 网络生成航路点平均总长度  : {avg_len_wp:.2f} 米 (生成成功率: {valid_wp_count}/{num_tests})")
    print(f"  3. 向量化 RRT 规划平均总长度 : {avg_len_vec:.2f} 米 (规划成功率: {success_vector_rrt_count}/{num_tests})") # ⭐ 新增
    print(f"  4. 标准 RRT* 规划平均总长度  : {avg_len_rrt:.2f} 米 (规划成功率: {success_rrt_count}/{num_tests})")
    print("=" * 60)

    