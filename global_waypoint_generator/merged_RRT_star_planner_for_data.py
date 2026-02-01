import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from env_generator_for_data import env_generator
from res_show_for_data import plot_tree_and_path, plot_tree_and_path_and_waypoints

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

def find_straight_waypoint(ori_path, env_map, epsilon=2.0, check_step=0.5, safety_margin=2.0):
    """
    输入:
        ori_path: 原始路径点序列, shape [N, 3]
        env_map:  环境地图数据 (map_dict)
        epsilon:  DP 算法阈值 (米)，建议设为 0.5 ~ 5.0，越小越精细，越大越稀疏
        check_step: 碰撞检测步长 (米)
        safety_margin: 安全余量 (米)，通常设为无人机半径或稍微大一点
    
    输出:
        straight_waypoints: list [[x,y,z], ...]
    """
    
    # 转换为 numpy array
    path = np.array(ori_path)
    if len(path) < 3:
        return path.tolist()

    # ==========================================
    # 0. 内部定义碰撞检测函数 (闭包)
    # ==========================================
    def is_line_collision_free(p1, p2, step_size, margin):
        """
        内部辅助函数：检测 p1-p2 连线是否安全
        直接使用外层的 env_map，不需要重复传参
        """
        obstacles = np.array(env_map['obstacles'])
        if len(obstacles) == 0: return True
            
        # 1. 提取障碍物信息
        obs_x = obstacles[:, 0]
        obs_y = obstacles[:, 1]
        obs_zmin = obstacles[:, 2]
        obs_zmax = obstacles[:, 3]
        # 核心：碰撞半径 = 物理半径 + 安全余量
        obs_r_squared = (obstacles[:, 4] + margin) ** 2 
        
        # 2. 线段采样
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6: return True
        
        n_steps = int(np.ceil(dist / step_size))
        t = np.linspace(0, 1, n_steps + 1)
        # [N_samples, 3]
        sample_points = p1 + np.outer(t, p2 - p1)
        
        # 3. 向量化检测
        sp_x = sample_points[:, 0:1] # [N, 1]
        sp_y = sample_points[:, 1:2]
        sp_z = sample_points[:, 2:3]
        
        # 广播对比: [N, 1] vs [M] (NumPy会自动广播为 [N, M])
        # 条件A: 高度碰撞
        collision_z = (sp_z >= obs_zmin) & (sp_z <= obs_zmax)
        
        # 条件B: 平面距离碰撞
        dist_sq = (sp_x - obs_x)**2 + (sp_y - obs_y)**2
        collision_xy = dist_sq <= obs_r_squared
        
        # 综合判定
        is_collided = np.any(collision_z & collision_xy)
        
        return not is_collided

    # ==========================================
    # 1. 贪婪视线剪枝 (Greedy LoS Pruning)
    # ==========================================
    pruned_path = [path[0]]
    current_idx = 0
    n_points = len(path)
    
    while current_idx < n_points - 1:
        found_next = False
        # 倒序查找最远的可见点
        for i in range(n_points - 1, current_idx, -1):
            # 这里的 check_step 和 safety_margin 使用了外层传入的参数
            if is_line_collision_free(path[current_idx], path[i], check_step, safety_margin):
                pruned_path.append(path[i])
                current_idx = i
                found_next = True
                break
        
        if not found_next:
            current_idx += 1
            pruned_path.append(path[current_idx])
            
    pruned_path = np.array(pruned_path)

    # ==========================================
    # 2. Douglas-Peucker (DP) 抽稀
    # ==========================================
    def point_line_distance(point, start, end):
        if np.all(start == end):
            return np.linalg.norm(point - start)
        line_vec = end - start
        point_vec = point - start
        cross_prod = np.cross(line_vec, point_vec)
        return np.linalg.norm(cross_prod) / np.linalg.norm(line_vec)

    def douglas_peucker(points, eps):
        if len(points) < 3: return points
        dmax = 0.0
        index = 0
        end = len(points) - 1
        
        # 寻找最远点
        # 优化：使用向量化计算点到直线距离可以更快，但循环写着简单，对于几百个点足够快
        for i in range(1, end):
            d = point_line_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        
        if dmax > eps:
            res1 = douglas_peucker(points[:index+1], eps)
            res2 = douglas_peucker(points[index:], eps)
            return np.vstack((res1[:-1], res2))
        else:
            return np.vstack((points[0], points[end]))

    final_waypoints = douglas_peucker(pruned_path, epsilon)
    
    return final_waypoints.tolist()


        
        
def plot_sample_scores(xyz, labels, best_path, waypoints, map_dim):
    """
    可视化生成的样本点及其得分
    :param xyz: [N, 3] 采样点的真实物理坐标
    :param labels: [N, 1] 每个点的得分 (0~1)
    :param best_path: [K, 3] 真实路径
    :param waypoints: [M, 3] 关键点
    :param map_dim: (Lx, Ly, Lz) 地图尺寸
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 绘制采样点 (散点图)
    # 技巧：过滤掉得分极低的点，或者让它们非常透明，否则会遮挡高分点
    # 这里我们根据分数设置颜色和大小
    
    # 展平 label
    scores = labels.flatten()
    
    # 颜色映射
    p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                   c=scores,              # 颜色深浅代表分数
                   cmap='jet',            # 蓝-青-黄-红 (红为高分)
                   s=scores*50 + 2,       # 大小也随分数变化，高分点更大
                   alpha=0.6,             # 透明度
                   label='Sampled Points')
    
    fig.colorbar(p, ax=ax, label='Score (Label)')

    # 2. 绘制真实路径 (黑色线条)
    path_arr = np.array(best_path)
    ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], 
            c='black', linewidth=3, label='Ground Truth Path')

    # 3. 绘制关键点 (红色星号)
    wp_arr = np.array(waypoints)
    ax.scatter(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2], 
               c='red', marker='*', s=200, label='Waypoints')

    # 4. 设置坐标轴
    ax.set_xlim(0, map_dim[0])
    ax.set_ylim(0, map_dim[1])
    ax.set_zlim(0, map_dim[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Data Sample Visualization: Points Score vs Ground Truth')
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
    alpha=1.0,      # <--- 建议这里默认改为 1.0，正如我们之前讨论的
    sigma1=0.05,    # 路段宽度 (归一化后)
    sigma2=0.02,    # 关键点精度 (归一化后)
    eps=1e-8,
    visualize=False
):
    """
    生成并保存一个训练样本 (.npz)
    
    修改说明:
    原 d_obs (最近障碍物距离) 已替换为 obs_normal (最近障碍物的法向量，3维)
    输出 points shape: [N, 8] -> [x, y, z, nx, ny, nz, f_start, f_goal]
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

    def get_nearest_obstacle_normal(point, obstacles):
        """
        计算点到最近障碍物的单位法向量 (指向远离障碍物的方向)
        返回: np.array([nx, ny, nz])
        """
        px, py, pz = point
        min_dist = float('inf')
        # 默认法向量（如果没有障碍物或出错），通常不会发生因为场景总有边界或物体
        # 这里给一个随机或者零向量都可以，但在有效范围内肯定会被覆盖
        best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        for (ox, oy, zmin, zmax, r_crash, _) in obstacles:
            # --- 1. 计算点相对于圆柱体的几何关系 ---
            dx = px - ox
            dy = py - oy
            d_xy = np.sqrt(dx**2 + dy**2) + eps # 防止除0
            
            # 水平方向的单位向量 (从圆心指向外)
            ux = dx / d_xy
            uy = dy / d_xy
            
            # 计算各维度到表面的有向距离 (positive means outside)
            dist_xy = d_xy - r_crash
            
            dist_z = 0.0
            vz = 0.0 # 垂直方向单位分量
            
            if pz > zmax:
                dist_z = pz - zmax
                vz = 1.0 # 在上方，向上指
            elif pz < zmin:
                dist_z = zmin - pz # 距离为正
                vz = -1.0 # 在下方，向下指
            else:
                dist_z = 0.0 # 在Z范围内
                vz = 0.0

            # --- 2. 确定该障碍物是否是最近的 ---
            # 我们需要计算欧氏距离来比较谁最近
            curr_dist = float('inf')
            
            # 分三种区域讨论距离
            if dist_xy > 0 and dist_z <= 0:   # 侧面区域
                curr_dist = dist_xy
            elif dist_xy <= 0 and dist_z > 0: # 上下底面区域 (圆柱盖子上方/下方)
                curr_dist = dist_z
            elif dist_xy > 0 and dist_z > 0:  # 角落区域 (斜上方/斜下方)
                curr_dist = np.sqrt(dist_xy**2 + dist_z**2)
            else: 
                # 理论上 valid points 不会进入内部 (check_collision 过滤了)
                # 但如果刚好在表面或数值误差，视为距离0
                curr_dist = 0.0
            
            # --- 3. 如果更近，更新法向量 ---
            if curr_dist < min_dist:
                min_dist = curr_dist
                
                # 计算该障碍物对该点的法向量
                nx, ny, nz = 0.0, 0.0, 0.0
                
                if dist_xy > 0 and dist_z <= 0:   # [侧面]: 法向量水平
                    nx, ny, nz = ux, uy, 0.0
                elif dist_xy <= 0 and dist_z > 0: # [顶底]: 法向量垂直
                    nx, ny, nz = 0.0, 0.0, vz
                elif dist_xy > 0 and dist_z > 0:  # [角落]: 向量合成
                    # 向量 = 水平分量 + 垂直分量
                    # 水平部分长度: dist_xy, 方向: (ux, uy)
                    # 垂直部分长度: dist_z,  方向: vz
                    vx = dist_xy * ux
                    vy = dist_xy * uy
                    vz_vec = dist_z * vz
                    
                    # 归一化
                    norm = np.sqrt(vx**2 + vy**2 + vz_vec**2) + eps
                    nx, ny, nz = vx/norm, vy/norm, vz_vec/norm
                else:
                    # 极其罕见的情况（在内部），默认为水平向外推
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
    
    # --- [修改核心] 计算最近障碍物的法向量 ---
    # 输出 shape: [N, 3]
    obs_normals = np.array([get_nearest_obstacle_normal(p, obstacles) for p in xyz], dtype=np.float32)
    
    d_s = np.linalg.norm(xyz - S[None], axis=1)
    d_g = np.linalg.norm(xyz - G[None], axis=1)
    denom = d_s + d_g + eps
    f_start = d_g / denom
    f_goal  = d_s / denom
    
    # --- [修改核心] 堆叠特征 ---
    # 现在维度变成了 8: [x, y, z, nx, ny, nz, f_start, f_goal]
    points = np.stack([
        xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2], # Index 0-2: 坐标
        obs_normals[:, 0], obs_normals[:, 1], obs_normals[:, 2], # Index 3-5: 障碍物法向量
        f_start,                                        # Index 6: Start特征
        f_goal                                          # Index 7: Goal特征
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
    # 7. 可视化模块 (如果需要)
    # ==========================================
    if visualize:
        print(f"[Visualizing] Plotting scores for {file_path}...")
        # 注意：这里的 visualize 函数可能需要适配新的 points 维度，
        # 但如果不画特征只画 xyz 和 labels，原函数应该可以用
        plot_sample_scores(xyz, labels, best_path, waypoints, env_map["map_dim"])



# 假设所有必要的函数 (env_generator, generate_valid_tasks, RRTStar, find_straight_waypoint, save_sample, plot_...) 都已经导入定义好了

if __name__ == '__main__':
    # ================= 配置区域 =================
    NUM_MAPS = 10          # 地图数量
    TASKS_PER_MAP = 200    # 每个地图的任务数
    BASE_SEED = 39         # 基础随机种子
    
    # 保存路径 (使用 raw string r"..." 防止转义错误)
    SAVE_DIR = r"C:\Users\Administrator\Nutstore\1\科研\科研具体idea实现进程\代码\idea1_code\global_waypoint_generator\raw_model\train_data2"
    
    # RRT* 参数
    R_AGENT_CRASH = 1.2
    R_AGENT_RISK = 1.7
    MAX_ITER = 3000
    EXPAND_DIS = 30
    SEARCH_RADIUS = 150
    # ===========================================

    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"开始生成数据: {NUM_MAPS} 个地图 x {TASKS_PER_MAP} 个任务 = {NUM_MAPS * TASKS_PER_MAP} 条数据")

    # --- 外层循环：生成地图 ---
    for map_id in range(NUM_MAPS):
        current_seed = BASE_SEED + map_id  # 确保每个地图种子不同
        
        print(f"\n[{map_id+1}/{NUM_MAPS}] 正在生成第 {map_id} 号地图 (Seed={current_seed})...")
        
        # 1. 生成地图
        env_map = env_generator(
            rho=0.8, 
            map_dim=(1500, 1500, 240),
            r_crash_range=(30, 50),
            r_risk_range=(3, 7),
            zmax_range=(30, 240),
            max_iter=5000,
            seed=current_seed
        )
        obstacle_list = env_map["obstacles"]
        print(f"地图生成完毕，包含 {len(obstacle_list)} 个障碍物。")

        # 2. 生成该地图下的任务列表
        # 注意：这里也传入 seed 保证可复现，或者您可以去掉 seed 让其完全随机
        tasks = generate_valid_tasks(TASKS_PER_MAP, env_map, min_dist=1000, seed=current_seed) 
        # 注意：我将 min_dist 改为了 1000，因为 1500*1500 的地图很难找到大量距离 >1500 的点对，容易卡死。如果您坚持要 1500 请改回。

        # --- 内层循环：执行路径规划 ---
        print(f"{'Task ID':<15} | {'Status':<10} | {'Time (s)':<10} | {'Length (m)':<10}")
        print("-" * 55)

        for task_id, (start, goal) in enumerate(tasks):
            # 初始化 RRT*
            planner = RRTStar(
                start=start, 
                goal=goal, 
                R_crash=R_AGENT_CRASH, 
                R_risk=R_AGENT_RISK, 
                obstacle_list=obstacle_list, 
                rand_area=[[0, 0, 0],[env_map["map_dim"][0], env_map["map_dim"][1], env_map["map_dim"][2]]],
                expand_dis=EXPAND_DIS,
                max_iter=MAX_ITER,
                search_radius=SEARCH_RADIUS,
                search_until_max_iter=True
            )
            
            start_time = time.time()
            path = planner.planning()
            
            # 计算平滑/关键点
            straight_waypoints = None
            if path is not None:
                straight_waypoints = find_straight_waypoint(
                    path, 
                    env_map, 
                    epsilon=10, 
                    check_step=0.5, 
                    safety_margin=1
                )

            end_time = time.time()
            elapsed = end_time - start_time
            
            # 构造唯一文件名
            # 格式: map0_task0.npz, map0_task1.npz ... map9_task199.npz
            file_name = f"map{map_id}_task{task_id}.npz"
            full_save_path = os.path.join(SAVE_DIR, file_name)

            if path is not None and straight_waypoints is not None:
                plen = planner.calculate_path_length(path)
                
                print(f"M{map_id}_T{task_id:<8} | {'Success':<10} | {elapsed:<10.4f} | {plen:<10.4f}")
                
                # --- 保存数据 ---
                save_sample2(
                    env_map,
                    file_path=full_save_path, # 传入完整路径
                    best_path=path,
                    waypoints=straight_waypoints,
                    N_attempts=4096,
                    alpha=0.4,
                    sigma1=0.3,
                    sigma2=0.2,
                    eps=1e-8,
                    visualize=False
                )
                
                # --- ⚠️ 警告：批量生成时请注释掉绘图，否则会弹出2000个窗口或内存溢出 ---
                # if task_id < 2: # 仅查看每个地图的前2个任务以检查效果
                #     plot_tree_and_path_and_waypoints(env_map, planner.node_list, path, straight_waypoints)

            else:
                print(f"M{map_id}_T{task_id:<8} | {'Failed':<10} | {elapsed:<10.4f} | {'N/A':<10}")

    print("\n所有数据生成任务结束！")
