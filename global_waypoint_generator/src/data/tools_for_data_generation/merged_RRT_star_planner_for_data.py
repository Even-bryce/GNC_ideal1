import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from env_generator_for_data import env_generator
from res_show_for_data import plot_tree_and_path, plot_tree_and_path_and_waypoints
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# 定义 Node 类，用于表示树中的每个节点
class Node:
    def __init__(self, x, y, z):
        self.x = x              # 节点的 x 坐标
        self.y = y              # 节点的 y 坐标
        self.z = z              # 节点的 z 坐标
        self.parent = None      # 节点的父节点，用于回溯路径
        self.cost = 0.0         # 从起点到该节点的路径成本


# 定义 RRTStar 类，用于实现 RRT* 算法
'''
这个版本融合了rrt*相关成熟算法的机制，支持渐进最优性，包含：
1. informed 采样
2. 目标偏置采样
3. apf引导扩展
这个算法的成功率几乎是100%，初始路径的寻找也比较快，但是他的碰撞检测比较简单不鲁棒，且没有我们自己的创新机制，对于密集地图仍然存在穿模现象
'''
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
        r = self.search_radius * math.sqrt(math.log(nnode) / nnode)  # 动态调整搜索半径
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
            if z_min <= node.z <= z_max:
                dist_xy = math.hypot(node.x - cx, node.y - cy)
                if dist_xy <= (self.R_crash + obs_R_crash):
                    return True
        return False
    
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
    
    def check_edge_collision(self, n1, n2):
        """
        检查线段 n1->n2 是否与任何障碍的 crash 区相交（XY 投影 + Z 重叠）
        使用线段到圆心的最短距离与 crash 半径和比较。
        """
        for cx, cy, z_min, z_max, obs_R_crash, obs_R_risk in self.obstacle_list:
            # 如果 z 方向不重叠则跳过
            seg_z_min = min(n1.z, n2.z)
            seg_z_max = max(n1.z, n2.z)
            if seg_z_max < z_min or seg_z_min > z_max:
                continue

            # 计算线段到障碍中心 (cx,cy) 的最短距离（XY 平面）
            dist_xy = self.point_to_line_distance_xy(cx, cy, n1.x, n1.y, n2.x, n2.y)

            # 若最短距离小于等于 crash 判定阈值（agent + obs），视为碰撞
            if dist_xy <= (self.R_crash + obs_R_crash):
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



# ------------------测试用函数------------------
def check_point_validity(x, y, z, obstacle_list):
    """检查单个点是否在任何障碍物的 risk 范围内"""
    for cx, cy, z_min, z_max, R_crash, R_risk in obstacle_list:
        if z_min <= z <= z_max:
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist <= R_risk:  # 严格禁止进入 risk 区域
                return False
    return True

def generate_valid_tasks(num_tasks, env_map, min_dist=200.0, seed=None):
    """
    生成 valid 的 (start, goal) 对

    参数:
        num_tasks : 需要生成的任务数量
        env_map   : 地图信息
        min_dist  : 起点与终点的最小距离阈值 (m)
        seed      : 随机种子
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    tasks = []
    obs_list = env_map["obstacles"]
    x_size = env_map["map_dim"][0]
    y_size = env_map["map_dim"][1]
    z_size = env_map["map_dim"][2]
   
    
    while len(tasks) < num_tasks:
        # ===== 生成 Start =====
        sx = random.uniform(0, x_size)
        sy = random.uniform(0, y_size)
        sz = random.uniform(0, z_size)

        # ===== 生成 Goal =====
        gx = random.uniform(0, x_size)
        gy = random.uniform(0, y_size)
        gz = random.uniform(0, z_size)

        # ===== 有效性检查 =====
        if not check_point_validity(sx, sy, sz, obs_list):
            continue
        if not check_point_validity(gx, gy, gz, obs_list):
            continue

        # ===== 起终点距离约束 =====
        dist = math.sqrt(
            (sx - gx) ** 2 +
            (sy - gy) ** 2 +
            (sz - gz) ** 2
        )

        if dist < min_dist:
            continue

        tasks.append(([sx, sy, sz], [gx, gy, gz]))

    return tasks


def find_straight_waypoint(ori_path, env_map=None, epsilon=20.0, check_step=0.5, safety_margin=1.2, min_dist_ratio=0.1):
    """
    【宏观骨架提取版：双向视线剪枝 + DP抽稀 + 空间聚类】
    1. 双向视线剪枝 (Bidirectional LoS Pruning)：基于环境障碍物进行物理视线检测，剔除所有冗余的中间节点。
    2. Douglas-Peucker (DP)：对剪枝后的路径进一步做纯几何形状简化（应对微小曲折）。
    3. 空间聚类：合并距离过近的关键点 (采用相对地图尺度的比例)。
    """
    path = np.array(ori_path)
    if len(path) < 3:
        return path.tolist()

    obstacles = env_map["obstacles"] if env_map else []

    # ==========================================
    # [新增] 动态计算真实聚类物理距离
    # ==========================================
    if env_map and "map_dim" in env_map:
        # 如果提供了地图信息，直接取地图长宽高的最大值作为尺度
        scale = max(env_map["map_dim"])
    else:
        # 如果没提供地图，兜底方案：取当前整条路径在 XYZ 三个方向上的最大跨度
        scale = np.max(np.ptp(path, axis=0))
        if scale == 0: 
            scale = 1.0
            
    # 将比例转化为真实物理距离 (例如 1500 * 0.1 = 150.0 米)
    real_min_dist = min_dist_ratio * scale

    # ==========================================
    # 0. 内部辅助函数：纯物理碰撞检测
    # ==========================================
    def check_segment_collision(p1, p2):
        """复刻 RRT* 的 check_edge_collision 逻辑，检查 3D 线段是否与障碍物圆柱体碰撞"""
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        seg_z_min, seg_z_max = min(z1, z2), max(z1, z2)

        for cx, cy, z_min, z_max, obs_R_crash, obs_R_risk in obstacles:
            if seg_z_max < z_min or seg_z_min > z_max:
                continue

            apx, apy = cx - x1, cy - y1
            abx, aby = x2 - x1, y2 - y1
            ab_len_sq = abx**2 + aby**2

            if ab_len_sq == 0:
                dist_xy = math.hypot(apx, apy)
            else:
                t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
                closest_x = x1 + t * abx
                closest_y = y1 + t * aby
                dist_xy = math.hypot(cx - closest_x, cy - closest_y)

            if dist_xy <= (safety_margin + obs_R_crash):
                return True
        return False

    # ==========================================
    # 1. 双向视线剪枝 (Bidirectional LoS Pruning)
    # ==========================================
    def forward_los(p_list):
        pruned = [p_list[0]]
        curr_idx = 0
        while curr_idx < len(p_list) - 1:
            for next_idx in range(len(p_list) - 1, curr_idx, -1):
                if not check_segment_collision(p_list[curr_idx], p_list[next_idx]):
                    pruned.append(p_list[next_idx])
                    curr_idx = next_idx
                    break
            else:
                curr_idx += 1
                pruned.append(p_list[curr_idx])
        return np.array(pruned)

    def backward_los(p_list):
        pruned = [p_list[-1]]
        curr_idx = len(p_list) - 1
        while curr_idx > 0:
            for next_idx in range(0, curr_idx):
                if not check_segment_collision(p_list[curr_idx], p_list[next_idx]):
                    pruned.append(p_list[next_idx])
                    curr_idx = next_idx
                    break
            else:
                curr_idx -= 1
                pruned.append(p_list[curr_idx])
        pruned.reverse()
        return np.array(pruned)

    if len(obstacles) > 0:
        path_fwd = forward_los(path)
        path_bwd = backward_los(path)
        path = path_fwd if len(path_fwd) <= len(path_bwd) else path_bwd

    if len(path) < 3:
        return path.tolist()

    # ==========================================
    # 2. 纯几何 Douglas-Peucker (DP) 抽稀
    # ==========================================
    def point_line_distance(point, start, end):
        if np.all(start == end): return np.linalg.norm(point - start)
        line_vec = end - start
        cross_prod = np.cross(line_vec, point - start)
        return np.linalg.norm(cross_prod) / np.linalg.norm(line_vec)

    def douglas_peucker(points, eps):
        if len(points) < 3: return points
        dmax, index = 0.0, 0
        end = len(points) - 1
        for i in range(1, end):
            d = point_line_distance(points[i], points[0], points[end])
            if d > dmax:
                index, dmax = i, d
                
        if dmax > eps:
            res1 = douglas_peucker(points[:index+1], eps)
            res2 = douglas_peucker(points[index:], eps)
            return np.vstack((res1[:-1], res2))
        else:
            return np.vstack((points[0], points[end]))

    dp_waypoints = douglas_peucker(path, epsilon)

    # ==========================================
    # 3. 空间聚类去重 (使用动态计算出的 real_min_dist)
    # ==========================================
    if len(dp_waypoints) <= 2:
        return dp_waypoints.tolist()

    sparse_wps = [dp_waypoints[0]] # 锚点 1：起点绝不动
    
    i = 1
    while i < len(dp_waypoints) - 1:
        cluster = [dp_waypoints[i]]
        j = i + 1
        
        # 顺着路径找，距离小于 real_min_dist 的全打包
        while j < len(dp_waypoints) - 1:
            if np.linalg.norm(dp_waypoints[j] - dp_waypoints[j-1]) < real_min_dist:
                cluster.append(dp_waypoints[j])
                j += 1
            else:
                break
        
        if len(cluster) == 1:
            representative_pt = cluster[0]
        else:
            center_pt = np.mean(cluster, axis=0)
            representative_pt = min(cluster, key=lambda p: np.linalg.norm(p - center_pt))
            
        # 检查选出的代表点，是否离队伍里最后一个点（包括起点）太近
        if np.linalg.norm(representative_pt - sparse_wps[-1]) >= real_min_dist:
            sparse_wps.append(representative_pt)
            
        i = j 

    goal_pt = dp_waypoints[-1]
    
    # 检查倒数第一个中间点是否跟终点贴脸了
    if len(sparse_wps) > 1 and np.linalg.norm(sparse_wps[-1] - goal_pt) < real_min_dist:
        sparse_wps.pop()

    sparse_wps.append(goal_pt) # 锚点 2：终点绝不动

    return [pt.tolist() for pt in sparse_wps]

        
        
def plot_sample_scores(xyz, labels, best_path, waypoints, extracted_wps, map_dim):
    """
    可视化生成的样本点及其得分，同时对比真值航路点与聚类提取的航路点
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 绘制采样点 (散点图)
    scores = labels.flatten()
    p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                   c=scores, cmap='jet', s=scores*50 + 2, alpha=0.6, label='Sampled Points')
    fig.colorbar(p, ax=ax, label='Score (Label)')

    # 2. 绘制真实路径 (黑色线条)
    path_arr = np.array(best_path)
    ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], 
            c='black', linewidth=3, label='Ground Truth Path')

    # 3. 绘制真实的 GT 关键点 (红色星号)
    wp_arr = np.array(waypoints)
    ax.scatter(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2], 
               c='red', marker='*', s=200, edgecolor='black', label='GT Waypoints')

    # 4. 绘制聚类提取出的航路点 (青色三角形)
    if len(extracted_wps) > 0:
        ax.scatter(extracted_wps[:, 0], extracted_wps[:, 1], extracted_wps[:, 2], 
                   c='cyan', marker='^', s=250, edgecolor='black', label='Extracted Waypoints')

    # 设置真实物理比例显示
    ax.set_xlim(0, map_dim[0])
    ax.set_ylim(0, map_dim[1])
    ax.set_zlim(0, map_dim[2])
    ax.set_box_aspect((map_dim[0], map_dim[1], map_dim[2]))

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Score Distribution & Waypoint Extraction Verification')
    ax.legend()
    plt.show()

def save_sample(
    env_map,
    file_path,
    best_path,      # [K, 3] 真实最优路径的一系列点 (密集)
    waypoints,      # [M, 3] 真实路径的关键航路点 (稀疏，包含起点和终点)
    N_attempts=4096,
    alpha=0.4,      # 路径基础分权重
    sigma1=0.05,    # 路段宽度 (归一化后)
    sigma2=0.02,    # 关键点精度 (归一化后)
    eps=1e-8,
    visualize=False
):
    """
    生成并保存一个训练样本 (.npz)
    
    输入:
        best_path: 用于计算 d_line (到路径线段的距离)
        waypoints: 用于计算 d_point (到关键点的距离) 以及提取 Start/Goal
        waypoints[0] 应为 Start, waypoints[-1] 应为 Goal
    """
    
    # ==========================================
    # 内部辅助函数
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

    def point_to_segment_distance(P, A, B):
        """
        计算点 P 到线段 AB 的最短距离 (向量化实现)
        P: [N, 3], A: [3], B: [3]
        """
        AB = B - A
        AP = P - A
        
        # 投影系数 t = (AP . AB) / (AB . AB)
        ab_sq = np.dot(AB, AB) + eps
        t = np.dot(AP, AB) / ab_sq
        
        # 限制 t 在 [0, 1] 之间（线段内）
        t = np.clip(t, 0.0, 1.0)
        
        # 投影点
        Proj = A + t[:, np.newaxis] * AB
        
        # 距离
        return np.linalg.norm(P - Proj, axis=1)
    

    

    # ==========================================
    # 1. 预处理输入数据
    # ==========================================
    path_arr = np.asarray(best_path, dtype=np.float32)   
    wp_arr   = np.asarray(waypoints, dtype=np.float32)   
    
    S = wp_arr[0]
    G = wp_arr[-1]
    
    Lx, Ly, Lz = env_map["map_dim"]
    xyz_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    xyz_max = np.array([Lx, Ly, Lz], dtype=np.float32)
    obstacles = env_map["obstacles"]
    
    center = 0.5 * (xyz_min + xyz_max)
    scale = max(Lx, Ly, Lz)

    # ==========================================
    # 2. 批量采样 (One-Pass)
    # ==========================================
    candidates = np.random.uniform(xyz_min, xyz_max, size=(N_attempts, 3))
    valid_pts = []
    
    for p in candidates:
        if not check_collision(p, obstacles):
            valid_pts.append(p)

    if len(valid_pts) == 0:
        pts = wp_arr.copy()
    else:
        pts = np.asarray(valid_pts, dtype=np.float32)

    # ==========================================
    # 3. 组合最终点集
    # ==========================================
    # 这里 xyz 存储的是【真实物理坐标】，这一点对绘图很重要
    xyz = np.vstack([S[None], G[None], pts]) 
    N_real = xyz.shape[0]

    # ==========================================
    # 4. 构建输入特征 (Input Features)
    # ==========================================
    xyz_norm = (xyz - center) / (scale + eps)

    # print(max(xyz_norm[:,0]), max(xyz_norm[:,1]), max(xyz_norm[:,2]))
    # print(min(xyz_norm[:,0]), min(xyz_norm[:,1]), min(xyz_norm[:,2]))
    
    d_obs = np.array([get_min_distance_to_obstacles(p, obstacles) for p in xyz], dtype=np.float32)
    d_obs_norm = d_obs / (scale + eps)
    
    d_s = np.linalg.norm(xyz - S[None], axis=1)
    d_g = np.linalg.norm(xyz - G[None], axis=1)
    denom = d_s + d_g + eps
    f_start = d_g / denom
    f_goal  = d_s / denom
    
    points = np.stack([
        xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2],
        d_obs_norm, f_start, f_goal
    ], axis=1).astype(np.float32)

    # ==========================================
    # 5. 标签计算 (Label Generation)
    # ==========================================
    labels = np.zeros((N_real, 1), dtype=np.float32)
    
    # --- 计算 d_point ---
    dists_to_wps = np.linalg.norm(xyz[:, None, :] - wp_arr[None, :, :], axis=2)
    d_point = np.min(dists_to_wps, axis=1) 

    # --- 计算 d_line ---
    d_line = np.full(N_real, float('inf'), dtype=np.float32)
    
    for k in range(len(path_arr) - 1):
        A = path_arr[k]
        B = path_arr[k+1]
        d_segment = point_to_segment_distance(xyz, A, B)
        d_line = np.minimum(d_line, d_segment)

    # --- 标签混合公式 ---
    d_line_norm = d_line / scale
    d_point_norm = d_point / scale
    
    y_line = alpha * np.exp(- (d_line_norm ** 2) / (2 * sigma1**2))
    y_point = 1.0 * np.exp(- (d_point_norm ** 2) / (2 * sigma2**2))
    
    labels[:, 0] = np.maximum(y_line, y_point)
    labels[0, 0] = 1.0 
    labels[1, 0] = 1.0

    # ==========================================
    # 6. 保存
    # ==========================================
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez(file_path, points=points, labels=labels)
    
    # ==========================================
    # 7. [新增] 可视化模块
    # ==========================================
    if visualize:
        print(f"[Visualizing] Plotting scores for {file_path}...")
        plot_sample_scores(xyz, labels, best_path, waypoints, env_map["map_dim"])
    


def save_sample2(
    env_map,
    file_path,
    best_path,      # [K, 3] 真实最优路径的一系列点 (密集)
    waypoints,      # [M, 3] 真实路径的关键航路点 (稀疏，包含起点和终点)
    N_attempts=4096,
    alpha=1.0,      # 路径基础分权重
    sigma1=0.05,    # 路段宽度 (归一化后)
    sigma2=0.02,    # 关键点精度 (归一化后)
    eps=1e-8,
    visualize=False
):
    """
    生成并保存一个训练样本 (.npz)
    【统一特征维度】输出 points shape: [N, 9]
    索引分布: [0:x, 1:y, 2:z, 3:f_start, 4:f_goal, 5:d_obs_norm, 6:nx, 7:ny, 8:nz]
    """
    
    # ==========================================
    # 内部辅助函数
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

    def get_nearest_obstacle_normal(point, obstacles):
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

            curr_dist = float('inf')
            if dist_xy > 0 and dist_z <= 0:   
                curr_dist = dist_xy
            elif dist_xy <= 0 and dist_z > 0: 
                curr_dist = dist_z
            elif dist_xy > 0 and dist_z > 0:  
                curr_dist = np.sqrt(dist_xy**2 + dist_z**2)
            else: 
                curr_dist = 0.0
            
            if curr_dist < min_dist:
                min_dist = curr_dist
                nx, ny, nz = 0.0, 0.0, 0.0
                
                if dist_xy > 0 and dist_z <= 0:   
                    nx, ny, nz = ux, uy, 0.0
                elif dist_xy <= 0 and dist_z > 0: 
                    nx, ny, nz = 0.0, 0.0, vz
                elif dist_xy > 0 and dist_z > 0:  
                    vx, vy, vz_vec = dist_xy * ux, dist_xy * uy, dist_z * vz
                    norm = np.sqrt(vx**2 + vy**2 + vz_vec**2) + eps
                    nx, ny, nz = vx/norm, vy/norm, vz_vec/norm
                else:
                    nx, ny, nz = ux, uy, 0.0
                    
                best_normal = np.array([nx, ny, nz], dtype=np.float32)

        return best_normal

    def point_to_segment_distance(P, A, B):
        AB = B - A
        AP = P - A
        ab_sq = np.dot(AB, AB) + eps
        t = np.dot(AP, AB) / ab_sq
        t = np.clip(t, 0.0, 1.0)
        Proj = A + t[:, np.newaxis] * AB
        return np.linalg.norm(P - Proj, axis=1)

    # ==========================================
    # 1. 预处理输入数据
    # ==========================================
    path_arr = np.asarray(best_path, dtype=np.float32)   
    wp_arr   = np.asarray(waypoints, dtype=np.float32)   
    
    S = wp_arr[0]
    G = wp_arr[-1]
    
    Lx, Ly, Lz = env_map["map_dim"]
    xyz_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    xyz_max = np.array([Lx, Ly, Lz], dtype=np.float32)
    obstacles = env_map["obstacles"]
    
    center = 0.5 * (xyz_min + xyz_max)
    scale = max(Lx, Ly, Lz)

    # ==========================================
    # 2. 批量采样 (One-Pass)
    # ==========================================
    candidates = np.random.uniform(xyz_min, xyz_max, size=(N_attempts, 3))
    valid_pts = []
    
    for p in candidates:
        if not check_collision(p, obstacles):
            valid_pts.append(p)

    if len(valid_pts) == 0:
        pts = wp_arr.copy()
    else:
        pts = np.asarray(valid_pts, dtype=np.float32)

    # ==========================================
    # 3. 组合最终点集
    # ==========================================
    xyz = np.vstack([S[None], G[None], pts]) 
    N_real = xyz.shape[0]

    # ==========================================
    # 4. 构建输入特征 (Input Features)
    # ==========================================
    xyz_norm = (xyz - center) / (scale + eps)
    
    d_obs = np.array([get_min_distance_to_obstacles(p, obstacles) for p in xyz], dtype=np.float32)
    d_obs_norm = d_obs / (scale + eps)

    obs_normals = np.array([get_nearest_obstacle_normal(p, obstacles) for p in xyz], dtype=np.float32)
    
    d_s = np.linalg.norm(xyz - S[None], axis=1)
    d_g = np.linalg.norm(xyz - G[None], axis=1)
    denom = d_s + d_g + eps
    f_start = d_g / denom
    f_goal  = d_s / denom
    
    # 强制起终点在索引 3 和 4 ---
    points = np.stack([
        xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2],          # Index 0, 1, 2: 坐标 xyz
        f_start, f_goal,                                         # Index 3, 4: 起终点特征
        d_obs_norm,                                              # Index 5: 障碍物距离
        obs_normals[:, 0], obs_normals[:, 1], obs_normals[:, 2], # Index 6, 7, 8: 障碍物法向量
    ], axis=1).astype(np.float32)

    # ==========================================
    # 5. 标签计算 (Label Generation) - 剔除起终点光晕
    # ==========================================
    labels = np.zeros((N_real, 1), dtype=np.float32)
    
    # 只提取中间的航路点算高斯圆
    mid_wps = wp_arr[1:-1]
    if len(mid_wps) > 0:
        dists_to_mid_wps = np.linalg.norm(xyz[:, None, :] - mid_wps[None, :, :], axis=2)
        d_point = np.min(dists_to_mid_wps, axis=1)
        d_point_norm = d_point / scale
        y_point = 1.0 * np.exp(- (d_point_norm ** 2) / (2 * sigma2**2))
    else:
        y_point = np.zeros(N_real, dtype=np.float32)

    # 计算路径线段管状高斯
    d_line = np.full(N_real, float('inf'), dtype=np.float32)
    for k in range(len(path_arr) - 1):
        A = path_arr[k]
        B = path_arr[k+1]
        d_segment = point_to_segment_distance(xyz, A, B)
        d_line = np.minimum(d_line, d_segment)

    d_line_norm = d_line / scale
    y_line = alpha * np.exp(- (d_line_norm ** 2) / (2 * sigma1**2))
    
    # 融合标签
    labels[:, 0] = np.maximum(y_line, y_point)
    
    # 仅保留绝对起终点自身的满分 (0号是起点，1号是终点)
    labels[0, 0] = 1.0 
    labels[1, 0] = 1.0

    # ==========================================
    # 6. 保存
    # ==========================================
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez(file_path, points=points, labels=labels)
    
    if visualize:
        print(f"[Visualizing] Plotting scores for {file_path}...")
        plot_sample_scores(xyz, labels, best_path, waypoints, env_map["map_dim"])

def save_sample3(
    env_map,
    file_path,
    best_path,      # [K, 3] 真实最优路径的一系列点 (密集)
    waypoints,      # [M, 3] 真实路径的关键航路点 (稀疏，包含起点和终点)
    N_attempts=4096,
    alpha=1.0,      # 路径基础分权重
    sigma1=0.05,    # 路段宽度 (此时代表占地图比例，如 0.05 代表 5% 的地图跨度)
    sigma2=0.02,    # 关键点精度 (同上)
    eps=1e-8,
    visualize=False
):
    """
    生成并保存一个训练样本 (.npz) - 异向高斯自适应版本
    针对 Z 轴跨度远小于 XY 轴的情况，自动生成扁平的椭球状高斯标签。
    """
    
    # ==========================================
    # 内部辅助函数
    # ==========================================
    def extract_waypoints(points, scores, eps=0.15, peak_radius=0.15, z_weight=1.0):
        if len(points) == 0:
            return np.empty((0, 3))

        # ================================
        # ⭐ Step 0: Z方向加权（拉伸Z轴距离，切断上下层连通）
        # ================================
        points_scaled = points.copy()
        points_scaled[:, 2] *= z_weight

        # ================================
        # Step 1: 局部极大值 (寻找峰值候选点)
        # ================================
        tree = KDTree(points_scaled)
        peaks = []
        peak_scores = []

        for i, p in enumerate(points_scaled):
            # 找周围半径内的邻居
            idx = tree.query_ball_point(p, r=peak_radius)

            is_peak = True
            for j in idx:
                # 如果周围有比自己得分严格更高的点，那自己就不是局部的绝对波峰
                if scores[j] > scores[i]:
                    is_peak = False
                    break

            if is_peak:
                peaks.append(points[i])  # ⚠️ 记录原始坐标
                peak_scores.append(scores[i])

        if len(peaks) == 0:
            return np.empty((0, 3))

        peaks = np.array(peaks)
        peak_scores = np.array(peak_scores)

        # ================================
        # ⭐ Step 2: DBSCAN 聚类（融合相近的局部极值点）
        # ================================
        peaks_scaled = peaks.copy()
        peaks_scaled[:, 2] *= z_weight

        clustering = DBSCAN(eps=eps, min_samples=1).fit(peaks_scaled)
        labels = clustering.labels_

        waypoints = []

        for label in set(labels):
            if label == -1:
                continue

            mask = (labels == label)
            cluster_points = peaks[mask]
            cluster_scores = peak_scores[mask]

            # 在同属于一个波峰簇的候选点中，选得分最高的那一个作为最终航路点
            best_idx = np.argmax(cluster_scores)
            waypoints.append(cluster_points[best_idx])

        return np.array(waypoints)

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

    def get_nearest_obstacle_normal(point, obstacles):
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

            curr_dist = float('inf')
            if dist_xy > 0 and dist_z <= 0:   
                curr_dist = dist_xy
            elif dist_xy <= 0 and dist_z > 0: 
                curr_dist = dist_z
            elif dist_xy > 0 and dist_z > 0:  
                curr_dist = np.sqrt(dist_xy**2 + dist_z**2)
            else: 
                curr_dist = 0.0
            
            if curr_dist < min_dist:
                min_dist = curr_dist
                nx, ny, nz = 0.0, 0.0, 0.0
                
                if dist_xy > 0 and dist_z <= 0:   
                    nx, ny, nz = ux, uy, 0.0
                elif dist_xy <= 0 and dist_z > 0: 
                    nx, ny, nz = 0.0, 0.0, vz
                elif dist_xy > 0 and dist_z > 0:  
                    vx, vy, vz_vec = dist_xy * ux, dist_xy * uy, dist_z * vz
                    norm = np.sqrt(vx**2 + vy**2 + vz_vec**2) + eps
                    nx, ny, nz = vx/norm, vy/norm, vz_vec/norm
                else:
                    nx, ny, nz = ux, uy, 0.0
                    
                best_normal = np.array([nx, ny, nz], dtype=np.float32)

        return best_normal

    def point_to_segment_distance(P, A, B):
        AB = B - A
        AP = P - A
        ab_sq = np.dot(AB, AB) + eps
        t = np.dot(AP, AB) / ab_sq
        t = np.clip(t, 0.0, 1.0)
        Proj = A + t[:, np.newaxis] * AB
        return np.linalg.norm(P - Proj, axis=1)

    # ==========================================
    # 1. 预处理输入数据
    # ==========================================
    path_arr = np.asarray(best_path, dtype=np.float32)   
    wp_arr   = np.asarray(waypoints, dtype=np.float32)   
    
    S = wp_arr[0]
    G = wp_arr[-1]
    
    Lx, Ly, Lz = env_map["map_dim"]
    xyz_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    xyz_max = np.array([Lx, Ly, Lz], dtype=np.float32)
    obstacles = env_map["obstacles"]
    
    center = 0.5 * (xyz_min + xyz_max)
    scale = max(Lx, Ly, Lz)

    # ==========================================
    # 2. 批量采样 (One-Pass)
    # ==========================================
    candidates = np.random.uniform(xyz_min, xyz_max, size=(N_attempts, 3))
    valid_pts = []
    
    for p in candidates:
        if not check_collision(p, obstacles):
            valid_pts.append(p)

    if len(valid_pts) == 0:
        pts = wp_arr.copy()
    else:
        pts = np.asarray(valid_pts, dtype=np.float32)

    # ==========================================
    # 3. 组合最终点集
    # ==========================================
    xyz = np.vstack([S[None], G[None], pts]) 
    N_real = xyz.shape[0]

    # ==========================================
    # 4. 构建输入特征 (Input Features)
    # （这里保持不变，给网络喂的依然是真实的几何比例）
    # ==========================================
    xyz_norm = (xyz - center) / (scale + eps)
    
    d_obs = np.array([get_min_distance_to_obstacles(p, obstacles) for p in xyz], dtype=np.float32)
    d_obs_norm = d_obs / (scale + eps)

    obs_normals = np.array([get_nearest_obstacle_normal(p, obstacles) for p in xyz], dtype=np.float32)
    
    d_s = np.linalg.norm(xyz - S[None], axis=1)
    d_g = np.linalg.norm(xyz - G[None], axis=1)
    denom = d_s + d_g + eps
    f_start = d_g / denom
    f_goal  = d_s / denom
    
    points = np.stack([
        xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2],          # Index 0, 1, 2
        f_start, f_goal,                                         # Index 3, 4
        d_obs_norm,                                              # Index 5
        obs_normals[:, 0], obs_normals[:, 1], obs_normals[:, 2], # Index 6, 7, 8
    ], axis=1).astype(np.float32)

    # ==========================================
    # 5. 标签计算 (Label Generation) - ⭐核心修复：异向高斯
    # ==========================================
    labels = np.zeros((N_real, 1), dtype=np.float32)
    
    # 构建坐标缩放比例尺 [Lx, Ly, Lz]
    # 通过将点云除以这个比例尺，物理空间被拉伸成了 1x1x1 的标准魔方
    dim_scale = np.array([Lx, Ly, Lz], dtype=np.float32) + eps
    
    # 计算用于打标签的“变形坐标”
    xyz_ratio = xyz / dim_scale
    wp_ratio = wp_arr / dim_scale
    path_ratio = path_arr / dim_scale
    
    # --- 计算 d_point (航路点) ---
    mid_wps_ratio = wp_ratio[1:-1]
    if len(mid_wps_ratio) > 0:
        # 在变形空间里算距离
        dists_to_mid_wps = np.linalg.norm(xyz_ratio[:, None, :] - mid_wps_ratio[None, :, :], axis=2)
        d_point_ratio = np.min(dists_to_mid_wps, axis=1)
        # 注意：这里直接用 d_point_ratio，不需要再除以 scale 了
        y_point = 1.0 * np.exp(- (d_point_ratio ** 2) / (2 * sigma2**2))
    else:
        y_point = np.zeros(N_real, dtype=np.float32)

    # --- 计算 d_line (管状路径) ---
    d_line_ratio = np.full(N_real, float('inf'), dtype=np.float32)
    for k in range(len(path_ratio) - 1):
        A_ratio = path_ratio[k]
        B_ratio = path_ratio[k+1]
        # 在变形空间里算线段距离
        d_segment = point_to_segment_distance(xyz_ratio, A_ratio, B_ratio)
        d_line_ratio = np.minimum(d_line_ratio, d_segment)

    # 同样，直接使用变形空间算出的比例距离
    y_line = alpha * np.exp(- (d_line_ratio ** 2) / (2 * sigma1**2))
    
    # 融合标签
    labels[:, 0] = np.maximum(y_line, y_point)
    labels[0, 0] = 1.0 
    labels[1, 0] = 1.0

    # ==========================================
    # 6. 保存
    # ==========================================
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez(file_path, points=points, labels=labels)
    # 将真实航路点也保存进去，方便后续可视化和测试
    np.savez(file_path, points=points, labels=labels, waypoints=wp_arr)
    
    if visualize:
        print(f"[Visualizing] Plotting scores and testing extraction for {file_path}...")
        
        # 为了加速计算并去除底噪，我们只提取得分大于阈值的点进行聚类
        mask = labels[:, 0] > 0.7
        high_score_xyz = xyz[mask]
        high_score_labels = labels[mask, 0]
        
        if len(high_score_xyz) > 0:
            # 💡 核心：把坐标转换到“各向异性比例空间”里去聚类，和标签的生成域保持一致！
            high_score_ratio = high_score_xyz / dim_scale
            
            # 调用聚类算法
            extracted_wps_ratio = extract_waypoints(
                points=high_score_ratio, 
                scores=high_score_labels, 
                eps=0.15,        
                peak_radius=0.15,        # 如果调小了的话就航路点就比较多
                z_weight=1.0             # 已经在 ratio 空间里压扁了 Z 轴，这里无需再额外加权
            )
            
            # 将聚类出来的结果从比例空间还原回真实物理坐标
            if len(extracted_wps_ratio) > 0:
                extracted_wps_physical = extracted_wps_ratio * dim_scale
            else:
                extracted_wps_physical = np.empty((0, 3))
        else:
            extracted_wps_physical = np.empty((0, 3))
            
        # 传递给统一的绘图函数
        plot_sample_scores(xyz, labels, best_path, waypoints, extracted_wps_physical, env_map["map_dim"])



def is_path_meaningful(waypoints, min_angle_deg=15.0):
    """
    轨迹质量质检员：
    1. 检查是否有中间航路点。
    2. 检查轨迹是否过于平直 (最大转弯角度是否大于 min_angle_deg)。
    """
    # 规则 1：如果没有中间航路点（总点数 <= 2），直接判定为太直，淘汰
    if waypoints is None or len(waypoints) <= 2:
        return False
        
    waypoints = np.array(waypoints)
    max_turn_angle = 0.0
    
    # 规则 2：遍历所有中间拐点，计算转弯角度
    for i in range(1, len(waypoints) - 1):
        # 向量 1：上一个点指向当前点
        v1 = waypoints[i] - waypoints[i-1]
        # 向量 2：当前点指向下一个点
        v2 = waypoints[i+1] - waypoints[i]
        
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 == 0 or n2 == 0:
            continue
            
        # 计算两个向量的夹角
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        # 防止浮点数精度超限
        cos_theta = np.clip(cos_theta, -1.0, 1.0) 
        angle_deg = np.degrees(np.arccos(cos_theta))
        
        if angle_deg > max_turn_angle:
            max_turn_angle = angle_deg
            
    # 如果整条路径上最大的那个拐角都比设定的阈值小，说明是一条笔直的伪折线，淘汰
    if max_turn_angle < min_angle_deg:
        return False
        
    return True



# 假设所有必要的函数 (env_generator, generate_valid_tasks, RRTStar, find_straight_waypoint, save_sample, plot_...) 都已经导入定义好了

if __name__ == '__main__':
    # ================= 配置区域 =================
    NUM_MAPS = 50          # 地图数量
    TASKS_PER_MAP = 30    # 需要成功保存的有效任务数
    BASE_SEED = 39         # 基础随机种子
    
    # 保存路径
    SAVE_DIR = r"C:\Users\Administrator\Desktop\experiments\train_data5"
    
    # RRT* 参数
    R_AGENT_CRASH = 1.2
    R_AGENT_RISK = 1.7
    MAX_ITER = 3000
    EXPAND_DIS = 30
    SEARCH_RADIUS = 150
    
    # 质量控制参数
    MIN_TURN_ANGLE = 15.0  # 判定有效拐弯的最小角度阈值 (度)，小于这个视作平直路线
    # ===========================================

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"开始生成高质量数据: {NUM_MAPS} 个地图 x 目标 {TASKS_PER_MAP} 个有效任务 = {NUM_MAPS * TASKS_PER_MAP} 条数据")

    # --- 外层循环：生成地图 ---
    for map_id in range(NUM_MAPS):
        current_seed = BASE_SEED + map_id  
        
        print(f"\n[{map_id+1}/{NUM_MAPS}] 正在生成第 {map_id} 号地图 (Seed={current_seed})...")
        
        # 1. 生成地图
        env_map = env_generator(
            rho=random.uniform(0.7, 0.9),   # 数据更丰富不容易出现过拟合
            map_dim=(1500, 1500, 240),
            r_crash_range=(30, 50),
            r_risk_range=(3, 7),
            zmax_range=(30, 240),
            max_iter=5000,
            seed=current_seed
        )
        obstacle_list = env_map["obstacles"]
        print(f"地图生成完毕，包含 {len(obstacle_list)} 个障碍物。开始执行 RRT* 与质量筛选...")

        print(f"{'Task ID':<10} | {'Attempt':<10} | {'Status':<15} | {'Time (s)':<10} | {'Length (m)':<10}")
        print("-" * 65)

        # 2. 动态循环：直到保存了足够多的高质量数据
        saved_tasks = 0
        attempts = 0  # 记录总尝试次数
        
        while saved_tasks < TASKS_PER_MAP:
            attempts += 1
            
            # 动态生成 1 个任务。利用 attempts 作为增量改变 seed，确保每次生成不同的点对
            task = generate_valid_tasks(1, env_map, min_dist=1000, seed=current_seed + attempts)
            start, goal = task[0]
            
            # 初始化 RRT*
            planner = RRTStar(
                start=start, goal=goal, 
                R_crash=R_AGENT_CRASH, R_risk=R_AGENT_RISK, 
                obstacle_list=obstacle_list, 
                rand_area=[[0, 0, 0], [env_map["map_dim"][0], env_map["map_dim"][1], env_map["map_dim"][2]]],
                expand_dis=EXPAND_DIS, max_iter=MAX_ITER, search_radius=SEARCH_RADIUS,
                search_until_max_iter=True
            )
            
            start_time = time.time()
            path = planner.planning()
            
            # 提取航路点
            straight_waypoints = None
            if path is not None:
                straight_waypoints = find_straight_waypoint(
                    path, env_map, epsilon=10, check_step=0.5, safety_margin=1
                )

            end_time = time.time()
            elapsed = end_time - start_time
            
            # ==========================================
            # 核心拦截逻辑：质量筛选
            # ==========================================
            if path is None or straight_waypoints is None:
                print(f"T{saved_tasks:<8} | A{attempts:<8} | {'Failed (RRT)':<15} | {elapsed:<10.4f} | {'N/A':<10}")
                continue
                
            if not is_path_meaningful(straight_waypoints, min_angle_deg=MIN_TURN_ANGLE):
                print(f"T{saved_tasks:<8} | A{attempts:<8} | {'Filtered (Straight)':<15} | {elapsed:<10.4f} | {'N/A':<10}")
                continue
            
            # ==========================================
            # 通过筛选，执行保存
            # ==========================================
            plen = planner.calculate_path_length(path)
            print(f"T{saved_tasks:<8} | A{attempts:<8} | {'Success (Saved)':<15} | {elapsed:<10.4f} | {plen:<10.4f}")
            
            file_name = f"map{map_id}_task{saved_tasks}.npz"
            full_save_path = os.path.join(SAVE_DIR, file_name)
            
            save_sample3(
                env_map,
                file_path=full_save_path, 
                best_path=path,
                waypoints=straight_waypoints,
                N_attempts=4096,
                alpha=0.3,
                sigma1=0.3,
                sigma2=0.225,
                eps=1e-8,
                visualize=True if saved_tasks < 0 else False  # 仅可视化前10个高质量任务
            )
            
            if saved_tasks < 0: 
                plot_tree_and_path_and_waypoints(env_map, planner.node_list, path, straight_waypoints)
            
            # 成功保存一个，计数器加 1
            saved_tasks += 1

    print("\n所有高质量数据生成任务结束！")
