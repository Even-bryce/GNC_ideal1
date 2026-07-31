import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from env_generator_for_data import env_generator, env_generator_maze, env_generator_cluster
from res_show import plot_map_and_waypoint, plot_tree_and_path

# 定义 Node 类，用于表示树中的每个节点
class Node:
    def __init__(self, x, y, z):
        self.x = x              # 节点的 x 坐标
        self.y = y              # 节点的 y 坐标
        self.z = z              # 节点的 z 坐标
        self.parent = None      # 节点的父节点，用于回溯路径
        self.cost = 0.0         # 从起点到该节点的路径成本

# 定义 RRT 类，用于实现 RRT 算法
class VRRT_star_Bi_Informed_APF:
    def __init__(self, env_map, waypoints, R_crash, R_risk, obstacle_list, expand_dis=25, max_iter=1500, search_radius=110, search_until_max_iter=True):
        """
        初始化 RRT 算法的参数
        :param env_map: 环境地图
        :param waypoints: 航路点坐标列表 [[x1,y1,z1],[x2,y2,z2],...,[xN,yN,zN]]
        :param obstacle_list: 障碍物列表，每个障碍物为 [x, y, zmin, zmax, R_ob_crash, R_ob_risk]
        :param rand_area: 随机采样区域的范围 [min, max]
        :param expand_dis: 树扩展的步长
        :param max_iter: 最大迭代次数
        :param search_radius: 搜索邻近节点的半径
        :param R_crash: 飞行器碰撞半径
        :param R_risk: 飞行器风险半径
        """
        self.env_map = env_map
        self.waypoints = waypoints             # 航路点列表
        self.expand_dis = expand_dis           # 每次扩展的步长
        self.max_iter = max_iter               # 最大迭代次数
        self.obstacle_list = obstacle_list     # 存储障碍物列表
        self.nodes_list_a = []                 # 树节点列表，初始化为空列表
        self.nodes_list_b = []
        self.R_crash = R_crash                 # 本体碰撞半径
        self.R_risk = R_risk                   # 本体风险半径
        self.search_radius = search_radius     # 搜索邻近节点的半径
        self.search_until_max_iter = search_until_max_iter  # 是否持续搜索直到最大迭代次数

    def planning(self):
        """
        主规划函数，用于生成从起点到目标的路径
        返回首次找到的可行路径，和循环结束后找到的最优路径；否则返回 None
        """
        # 转换为 (n,3) 数组
        waypoints_array = np.array(self.waypoints)
        # 起点、终点列表 [[[x1,y1,z1]],[[x2,y2,z2]],...]
        node_array_a = waypoints_array[:-1, np.newaxis, :]
        node_array_b = waypoints_array[1:, np.newaxis, :]
        # 起点、终点Node类列表[[[start]], [[start*]], [[start**]], ...]
        self.nodes_list_a = [[Node(coord[0], coord[1], coord[2])] for coord in node_array_a[:, 0, :]]
        self.nodes_list_b = [[Node(coord[0], coord[1], coord[2])] for coord in node_array_b[:, 0, :]]
        
        num_trees = len(self.nodes_list_a)
        # 起点节点列表
        goal_nodes_list_a = []
        for i in range(num_trees):
            node = Node(waypoints_array[i, 0], waypoints_array[i, 1], waypoints_array[i, 2])
            node.parent = None
            node.cost = float('inf')
            goal_nodes_list_a.append(node)
        # 终点节点列表
        goal_nodes_list_b = []
        for i in range(1, num_trees + 1):
            node = Node(waypoints_array[i, 0], waypoints_array[i, 1], waypoints_array[i, 2])
            node.parent = None
            node.cost = float('inf')
            goal_nodes_list_b.append(node)
        
        first_path_found = np.full(num_trees, False, dtype=bool)
        first_path = np.full(num_trees, None, dtype=object)
        iteration_list = np.zeros(num_trees, dtype=int)
        time_first_list = [None] * num_trees
        path_length_first_list = [None] * num_trees
        
        best_paths = np.full(num_trees, None, dtype=object)
        best_costs = [np.inf] * num_trees
        
        start_time = time.time()
        
        for i in range(self.max_iter):  # 循环执行最大迭代次数
            node_array_a = self.build_node_array(self.nodes_list_a)
            node_array_b = self.build_node_array(self.nodes_list_b)
            # 随机采样
            random_nodes_array = self.sample_free_vectorized(waypoints_array)

            # 已找到路径的航段替换为椭球采样
            if self.search_until_max_iter:
                if all(first_path_found):
                    for j in range(num_trees):
                        # 获取起点、终点
                        start = waypoints_array[j]
                        goal = waypoints_array[j + 1]
                        # 使用当前最优成本作为椭球长轴参数
                        c_max = best_costs[j]
                        # 生成椭球内采样点并替换
                        random_nodes_array[j] = self._sample_informed_ellipsoid(start, goal, c_max)

            # 找到距离随机点最近的已有节点
            nearest_ind = self.get_nearest_node_index(node_array_a, random_nodes_array)
            nearest_nodes_list = [self.nodes_list_a[i][nearest_ind[i]] for i in range(len(nearest_ind))]

            # 计算扩展方向并生成新节点
            new_nodes_list = self.apf_steer(nearest_nodes_list, random_nodes_array, goal_nodes_list_b)
            
            # 找到距离新节点最近的另一棵树中的节点
            new_nodes_array = np.array([[node.x, node.y, node.z] for node in new_nodes_list])
            nearest_connect_ind = self.get_nearest_node_index(node_array_b, new_nodes_array)
            nearest_connect_nodes_list = [self.nodes_list_b[i][nearest_connect_ind[i]] for i in range(len(nearest_connect_ind))]
            # 碰撞检测：新节点——连接节点
            goal_collision_results = self.check_collision_vectorized(new_nodes_list, nearest_connect_nodes_list)
            
            # 寻找临近节点索引
            near_inds = self.find_near_nodes_vectorized(new_nodes_list, node_array_a)
            # 选择最佳父节点（已包含碰撞检测）
            new_nodes_list = self.choose_best_parent(new_nodes_list, nearest_nodes_list, near_inds)
            
            for j, new_node in enumerate(new_nodes_list):
                # 有最佳父节点，表明无碰撞，将新节点加入树
                if new_node.parent is not None:
                    self.nodes_list_a[j].append(new_node)
                    # 重连接
                    self.rewire(new_node, near_inds[j], self.nodes_list_a[j])
                    
                    # 检查是否可直接连接到另一棵树的最近节点
                    if goal_collision_results[j] and self.calc_distance(new_node, nearest_connect_nodes_list[j]) < 10 * self.expand_dis:
                        # 生成路径
                        root_a = self.nodes_list_a[j][0]
                        start_pt = waypoints_array[j]
                        if (root_a.x, root_a.y, root_a.z) == (start_pt[0], start_pt[1], start_pt[2]):
                            # 当前树是起点树
                            path_forward = self.generate_final_path_from_node(new_node)
                            path_backward = self.generate_final_path_from_node(nearest_connect_nodes_list[j])[::-1]
                            
                        else:
                            # 当前树是终点树，交换路径顺序
                            path_forward = self.generate_final_path_from_node(nearest_connect_nodes_list[j])
                            path_backward = self.generate_final_path_from_node(new_node)[::-1]
                        full_path = path_forward + path_backward
                        path_cost = calculate_path_length(full_path)
                        
                        # 首次找到路径
                        if not first_path_found[j]:
                            first_path_found[j] = True
                            time_first_list[j] = time.time() - start_time
                            iteration_list[j] = i
                            first_path[j] = full_path
                            path_length_first_list[j] = path_cost

                            best_costs[j] = path_length_first_list[j]
                            best_paths[j] = first_path[j]
                            
                        if self.search_until_max_iter and first_path_found[j]:
                            if path_cost < best_costs[j]:
                                best_costs[j] = path_cost
                                best_paths[j] = full_path
                                
            if all(first_path_found):
                # 合并所有航路段
                first_combined_path = []
                for seg in first_path:
                    if first_combined_path and first_combined_path[-1] == seg[0]:
                        first_combined_path.extend(seg[1:])
                    else:
                        first_combined_path.extend(seg)
                        
                if not self.search_until_max_iter:
                    return first_path_found, time_first_list, iteration_list, path_length_first_list, path_length_first_list, first_combined_path, first_combined_path
            
            self.nodes_list_a, self.nodes_list_b = self.nodes_list_b, self.nodes_list_a
            goal_nodes_list_a, goal_nodes_list_b = goal_nodes_list_a, goal_nodes_list_b

        if all(best_paths):
            # 用剩余迭代次数优化后的路径
            final_combined_path = []
            for seg in best_paths:
                if final_combined_path and final_combined_path[-1] == seg[0]:
                    final_combined_path.extend(seg[1:])
                else:
                    final_combined_path.extend(seg)

            return first_path_found, time_first_list, iteration_list, path_length_first_list, best_costs, first_combined_path, final_combined_path
        
        return None, None, None, None, None, None, None

    def _sample_informed_ellipsoid(self, start, goal, c_max):
        """
        在以 start 和 goal 为焦点、c_max 为椭圆长轴的椭球内均匀采样一个点。
        当 c_max 接近两焦点距离时退化为线段采样。
        """
        start = np.array(start)
        goal = np.array(goal)
        d = np.linalg.norm(goal - start)
        if c_max <= d:
            # 退化情况：椭球退化为线段，直接在线段上随机采样
            t = np.random.uniform(0, 1)
            return start + t * (goal - start)

        # 椭球中心
        center = (start + goal) / 2.0
        # 焦点半距
        c_foci = d / 2.0
        # 长半轴
        a = c_max / 2.0
        # 短半轴
        b = np.sqrt(a**2 - c_foci**2)

        # 建立局部坐标系：x 轴指向 goal-start 方向
        dir_vec = (goal - start) / d
        # 构造两个正交方向（任意但与 dir_vec 正交）
        if abs(dir_vec[0]) > 1e-6 or abs(dir_vec[1]) > 1e-6:
            u2 = np.array([-dir_vec[1], dir_vec[0], 0.0])
        else:
            u2 = np.array([1.0, 0.0, 0.0])
        u2 = u2 / np.linalg.norm(u2)
        u3 = np.cross(dir_vec, u2)
        u3 = u3 / np.linalg.norm(u3)
        # 旋转矩阵：列向量为局部坐标系的基
        L = np.column_stack((dir_vec, u2, u3))
        # 缩放矩阵
        S = np.diag([a, b, b])

        # 在单位球内均匀采样
        # 随机方向
        dir_random = np.random.randn(3)
        dir_random = dir_random / np.linalg.norm(dir_random)
        # 半径按体积分布：r = U(0,1)^{1/3}
        r = np.cbrt(np.random.uniform(0, 1))
        x_ball = dir_random * r

        # 变换到椭球坐标
        sample = center + L @ (S @ x_ball)
        return sample
    
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
        
        # 直接从 mins 和 maxs 得到扩展后的范围
        mins_expanded = mins.copy()
        maxs_expanded = maxs.copy()
        mins_expanded[:, :2] -= 5 * self.expand_dis
        maxs_expanded[:, :2] += 5 * self.expand_dis

        # 裁剪到地图边界（假设 self.map_dim = [Lx, Ly, Lz]）
        Lx, Ly, _ = self.env_map["map_dim"][0], self.env_map["map_dim"][1], self.env_map["map_dim"][2]
        mins_expanded[:, :2] = np.clip(mins_expanded[:, :2], 0, [Lx, Ly])
        maxs_expanded[:, :2] = np.clip(maxs_expanded[:, :2], 0, [Lx, Ly])
        
        # 生成 [0,1) 间的 (n-1,3) 随机数组
        n_segments = len(starts)
        
        rnd_gen = np.random.default_rng()
        random_points = rnd_gen.random((n_segments, 3))
        
        # 缩放得到采样点
        samples = mins + random_points * (maxs_expanded - mins_expanded)
        
        return samples
        
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
    
    def apf_steer(self, from_nodes, to_nodes_array, goal_nodes_list):
        """
        从 from_nodes 向合力方向扩展新节点，合力 = 指向采样点的引力 + 指向终点的引力 + 障碍物斥力
        :param from_nodes: 起始节点列表
        :param to_nodes_array: 随机采样点数组 (N-1, 3)
        :return: 新节点列表
        """
        from_coords = np.array([[node.x, node.y, node.z] for node in from_nodes], dtype=float)
        to_nodes_array = np.asarray(to_nodes_array, dtype=float)
        goal_coords = np.array([[node.x, node.y, node.z] for node in goal_nodes_list], dtype=float)

        # 采样点引力方向
        rand_dir = to_nodes_array - from_coords
        rand_dist = np.linalg.norm(rand_dir, axis=1, keepdims=True)
        rand_unit = np.where(rand_dist > 0, rand_dir / rand_dist, 0.0)

        # 目标点引力方向
        goal_dir = goal_coords - from_coords
        goal_dist = np.linalg.norm(goal_dir, axis=1, keepdims=True)
        goal_unit = np.divide(goal_dir, goal_dist, where=goal_dist > 0, out=np.zeros_like(goal_dir))

        # 障碍物斥力
        repulsion = np.zeros_like(from_coords)
        for i, node in enumerate(from_nodes):
            fx, fy = 0.0, 0.0
            for obs in self.obstacle_list:
                xc, yc, zmin, zmax, r_crash, r_risk = obs
                if node.z < zmin or node.z > zmax:
                    continue
                dx = node.x - xc
                dy = node.y - yc
                dist_h = math.hypot(dx, dy)
                if dist_h <= r_risk and r_risk > 0:
                    mag = (r_risk - dist_h) / r_risk
                    if dist_h > 1e-6:
                        dir_x = dx / dist_h
                        dir_y = dy / dist_h
                    else:
                        dir_x, dir_y = 0.0, 0.0
                    fx += mag * dir_x
                    fy += mag * dir_y
            repulsion[i, 0] = fx
            repulsion[i, 1] = fy

        # 合力
        total_force = rand_unit + 0.3 * goal_unit + repulsion
        force_norm = np.linalg.norm(total_force, axis=1, keepdims=True)
        unit_dir = np.where(force_norm > 0, total_force / force_norm, rand_unit)

        # 扩展步长
        new_coords = from_coords + unit_dir * self.expand_dis

        # 创建新节点
        new_nodes = []
        for i in range(len(from_nodes)):
            new_node = Node(new_coords[i, 0], new_coords[i, 1], new_coords[i, 2])
            new_node.parent = from_nodes[i]
            actual_dist = self.calc_distance(from_nodes[i], new_node)
            new_node.cost = from_nodes[i].cost + actual_dist
            new_nodes.append(new_node)
        return new_nodes
    
    def find_near_nodes_vectorized(self, new_nodes_list, node_array):
        """
        找到新节点附近的节点索引
        :param new_nodes_list: 新节点列表
        :param node_array: 所有节点的坐标数组 (N-1, M, 3)
        :return: 附近节点的索引列表near_nodes_indices_list
        """
        # 提取新节点坐标 (N-1, 3)
        new_coords = np.array([[node.x, node.y, node.z] for node in new_nodes_list])
        # 扩展维度 (N-1, 1, 3)
        new_expanded = new_coords[:, np.newaxis, :]
        # 计算距离平方矩阵 (N-1, M)
        diff = node_array - new_expanded
        distances_sq = np.sum(diff ** 2, axis=2)
        
        r = self.search_radius
        r_sq = r * r
        
        near_nodes_indices_list = []
        for i in range(distances_sq.shape[0]):
            # 距离 <= r 的索引
            indices = np.where(distances_sq[i] <= r_sq)[0].tolist()
            near_nodes_indices_list.append(indices)
        
        return near_nodes_indices_list
    
    def choose_best_parent(self, new_nodes_list, nearest_nodes_list, near_nodes_indices_list):
        """
        选择最佳父节点
        :param new_nodes_list: 新节点列表
        :param nearest_nodes_list: 最近节点列表
        :param near_nodes_indices_list: 附近节点的索引列表
        :return: 更新后的新节点
        """
        for i, new_node in enumerate(new_nodes_list):
            node_list = self.nodes_list_a[i]
            # 候选父节点
            candidates = set()
            candidates.add(nearest_nodes_list[i])
            for idx in near_nodes_indices_list[i]:
                candidates.add(node_list[idx])
            
            # 候选父节点的路径成本（“起点——候选父节点——新节点”）
            candidate_costs = []
            for node in candidates:
                dist = self.calc_distance(node, new_node)
                total_cost = node.cost + dist
                candidate_costs.append((total_cost, node))
            
            # 按路径成本升序
            candidate_costs.sort(key=lambda x: x[0])
            
            # 依次进行碰撞检测
            best_parent = None
            min_cost = float('inf')
            for cost, node in candidate_costs:
                if not self.check_edge_collision(new_node, node):
                    best_parent = node
                    min_cost = cost
                    break
            if best_parent is not None:
                new_node.cost = min_cost
                new_node.parent = best_parent
            else:
                # 无法找到无碰撞父节点
                new_node.parent = None
                new_node.cost = float('inf')
        return new_nodes_list
    
    def calc_distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 + (node1.z - node2.z) ** 2)
    
    def rewire(self, new_node, near_inds, node_list):
        """
        重新连接邻近节点以优化路径
        :param new_node: 新节点
        :param near_inds: 附近节点的索引
        :param node_list: 当前树的所有节点列表
        """
        for idx in near_inds:
            near_node = node_list[idx]
            if near_node is new_node:
                continue
            
            # 计算“起点——新节点——临近节点”的路径成本
            new_cost = new_node.cost + self.calc_distance(new_node, near_node)
            
            # 路径成本减少，则进行碰撞检测
            if new_cost < near_node.cost:
                if not self.check_edge_collision(new_node, near_node):
                    # 更新节点关系
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    self.propagate_cost_to_leaves(near_node, node_list)
    
    def propagate_cost_to_leaves(self, parent_node, node_list):
        '''
        递归更新子节点的成本
        '''
        for node in node_list:
            if node.parent is parent_node:
                node.cost = self.calc_distance(parent_node, node) + parent_node.cost
                self.propagate_cost_to_leaves(node, node_list)
    
    def check_edge_collision(self, node_a, node_b):
        """
        检测线段是否与任何障碍物碰撞。
        返回 True 表示碰撞，False 表示无碰撞。
        """
        # 线段包围盒
        x1, y1, z1 = node_a.x, node_a.y, node_a.z
        x2, y2, z2 = node_b.x, node_b.y, node_b.z
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        min_z = min(z1, z2)
        max_z = max(z1, z2)

        # 候选障碍物
        candidates = []
        for obs in self.obstacle_list:
            xc, yc, zmin, zmax, r_crash, _ = obs
            # 圆柱包围盒
            obs_min_x = xc - r_crash
            obs_max_x = xc + r_crash
            obs_min_y = yc - r_crash
            obs_max_y = yc + r_crash
            if (max_x < obs_min_x or min_x > obs_max_x or
                max_y < obs_min_y or min_y > obs_max_y or
                max_z < zmin or min_z > zmax):
                continue
            candidates.append(obs)
        if not candidates:
            return False

        # 采样检测
        length = math.hypot(x2 - x1, y2 - y1, z2 - z1)
        # 采样步长
        step = getattr(self, 'collision_check_resolution', 2)
        # 最大采样点数
        max_samples = 100
        num_samples = max(2, min(int(length / step) + 1, max_samples))

        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.0
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            pz = z1 + t * (z2 - z1)
            for (xc, yc, zmin, zmax, r_crash, _) in candidates:
                dx = px - xc
                dy = py - yc
                # 碰撞
                if dx*dx + dy*dy <= r_crash*r_crash and zmin <= pz <= zmax:
                    return True

        return False
    
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
# 主程序
# ==========================================
if __name__ == '__main__':
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
    # plot_map_and_waypoint(env_map, waypoints)
    
    # 设定 RRT* 参数
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    env_results = []
    
    # 规划次数
    num_of_tests = 10
    # 测评指标
    success_count = 0
    total_time_first = []      # 首次找到路径的总耗时
    total_time_final = []      # 最终找到路径的总耗时
    total_iter_needed = []     # 完成规划时，所有航路段的最大迭代次数
    total_length_first = []    # 首次规划的路径总长度
    total_length_final = []    # 最终规划的路径总长度
    
    time_segments = []          # 每个航段的规划时间
    length_segments = []        # 每个航段的首次长度
    length_segments_final = []  # 每个航段的最终长度
    
    for j in range(num_of_tests):
        # 初始化 RRT*
        print(f"\n测试 #{j + 1}")
        rrt_star = VRRT_star_Bi_Informed_APF(
            env_map=env_map,
            waypoints=waypoints,
            R_crash=r_agent_crash, 
            R_risk=r_agent_risk, 
            obstacle_list=obstacle_list, 
            expand_dis=10,
            search_radius=30,
            max_iter=2000,
            search_until_max_iter=True
        )
        start_time = time.time()
        first_path_found, time_first, iteration_find_path, path_length_list, path_length_final, first_path, final_best_path = rrt_star.planning()
        end_time = time.time()
        
        # 打印单次结果
        num_segments = len(first_path_found)
        print("-" * 55)
        print(f"{'航段':<2} | {'状态':<2} | {'迭代轮次':<3} | {'规划时间':<4} | {'首次长度':<4} | {'最终长度':<4}")
        print("-" * 55)
        for i in range(num_segments):
            success = first_path_found[i]
            status = "成功" if success else "失败"
            # 迭代轮次
            iter_val = str(iteration_find_path[i]) if success else "N/A"
            # 规划时间
            time_val = f"{time_first[i]:.3f}" if success and time_first[i] is not None else "N/A"
            # 路径长度
            length_val = f"{path_length_list[i]:.1f}" if success else "N/A"
            # 最终长度
            length_final = f"{path_length_final[i]:.1f}" if success else "N/A"
            # 航段编号
            print(f"{i+1:<4} | {status:<2} | {iter_val:<8} | {time_val:<8} | {length_val:<8} | {length_final:<8}")
        print("-" * 55)
        
        all_success = all(first_path_found)
        
        if all_success:
            max_iter_needed = np.max(iteration_find_path)
            total_time = end_time - start_time
            total_length = sum([path_length_list[i] for i in range(len(path_length_list)) if first_path_found[i]])
            print(f"最终耗时：{total_time:.3f}s")
            print(f"首次长度：{total_length:.2f}")
            # plot_tree_and_path(env_map, rrt_star.nodes_list_a, final_best_path, waypoints)
            success_count += 1
            total_time_first.append(total_time)
            total_iter_needed.append(max_iter_needed)
            total_length_first.append(total_length)
            
            time_segments.append(time_first)
            length_segments.append(path_length_list)
            length_segments_final.append(path_length_final)
            print(f"当前平均耗时：{sum(total_time_first) / success_count:.3f}s")
            
    print("=" * 30)
    if success_count > 0:
        avg_success_rate = (success_count / num_of_tests) * 100
        avg_time_first = sum(total_time_first) / success_count
        avg_iter = sum(total_iter_needed) / success_count
        avg_length_first = sum(total_length_first) / success_count
        
        print(f"Bi-VRRT*: 步长 {rrt_star.expand_dis}, 搜索半径{rrt_star.search_radius}")
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
        average_times = []
        average_lengths_first = []
        average_lengths_final = []
        for i in range(num_segments):
            # 平均首次时间
            segment_times = [trial[i] for trial in time_segments]
            avg_time = sum(segment_times) / len(segment_times)
            average_times.append(avg_time)
            
            # 平均首次长度
            segment_lengths_first = [trial[i] for trial in length_segments]
            avg_len_first = sum(segment_lengths_first) / len(segment_lengths_first)
            average_lengths_first.append(avg_len_first)
            
            # 平均最终长度
            segment_lengths_final = [trial[i] for trial in length_segments_final]
            avg_len_final = sum(segment_lengths_final) / len(segment_lengths_final)
            average_lengths_final.append(avg_len_final)

        print("-" * 60)
        print(f"{'航段':<4} | {'首次时间':<4} | {'首次长度':<4} | {'最终长度':<4} | {'优化比例':<4}")
        print("-" * 60)
        for idx in range(num_segments):
            avg_t = average_times[idx]
            avg_f = average_lengths_first[idx]
            avg_ff = average_lengths_final[idx]
            # 计算优化比例
            if avg_f != 0:
                improvement = (avg_f - avg_ff) / avg_f * 100
            else:
                improvement = 0.0
            print(f"{idx+1:<6} | {avg_t:<8.3f} | {avg_f:<8.3f} | {avg_ff:<8.3f} | {improvement:<6.2f}%")
        print("-" * 60)