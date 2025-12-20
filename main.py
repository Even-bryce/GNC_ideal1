import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from env_generator import env_generator
from res_show import plot_map, plot_tree_and_path
from path_optimizer import PathOptimizer

# 定义 Node 类，用于表示树中的每个节点
class Node:
    def __init__(self, x, y, z):
        self.x = x              # 节点的 x 坐标
        self.y = y              # 节点的 y 坐标
        self.z = z              # 节点的 z 坐标
        self.parent = None      # 节点的父节点，用于回溯路径
        self.cost = 0.0         # 从起点到该节点的路径成本


# 定义 RRTStar 类，用于实现 RRT* 算法
class RRTStar:
    def __init__(self, start, goal, R_crash, R_risk, obstacle_list, rand_area, expand_dis=30, max_iter=1500, search_radius=150, safety_margin=0.3, search_until_max_iter=True):
        """
        初始化 RRT* 算法的参数
        :param start: 起点坐标 [x, y, z]
        :param goal: 目标坐标 [x, y, z]
        :param obstacle_list: 障碍物列表，每个障碍物为 [x, y, zmin, zmax, R_ob_crash, R_ob_risk]
        :param rand_area: 随机采样区域的范围 [min, max]
        :param expand_dis: 树扩展的步长
        :param max_iter: 最大迭代次数
        :param search_radius: 搜索邻近节点的半径
        :param R_crash: 飞行器碰撞半径
        :param R_risk: 飞行器风险半径
        """
        self.start = Node(start[0], start[1], start[2])  # 创建起点节点
        self.goal = Node(goal[0], goal[1], goal[2])     # 创建目标节点
        self.min_rand = rand_area[0]           # 随机采样区域的最小值
        self.max_rand = rand_area[1]           # 随机采样区域的最大值
        self.z_rand = rand_area[2]          # 随机采样区域的最大z值
        self.expand_dis = expand_dis           # 每次扩展的步长
        self.max_iter = max_iter               # 最大迭代次数
        self.obstacle_list = obstacle_list     # 存储障碍物列表
        self.node_list_start = [self.start]          # 起始点为根的树节点列表，初始化时只包含起点
        self.node_list_goal = [self.goal]            # 目标点为根的树节点列表，初始化时只包含目标点
        self.merged_node_list = []      # 合并后的节点列表
        self.search_radius = search_radius     # 搜索邻近节点的半径
        self.R_crash = R_crash   # 本体碰撞半径
        self.R_risk = R_risk     # 本体风险半径
        self.search_until_max_iter = search_until_max_iter  # 是否持续搜索直到最大迭代次数
        self.first_path_found = False  # 是否已找到首条路径
        self.c_best = float("inf")             # 当前最佳路径成本
        self.c_min = self.calc_dist_to_goal(self.start.x, self.start.y, self.start.z)
        self.use_informed_sampling = False     # 是否启用 Informed 采样
        self.safety_margin = safety_margin     # 碰撞检测安全阈值
        

    def planning(self):

        self.goal.cost = 0.0
        self.goal.parent = None

        start_time = time.time()

        tree1 = self.node_list_start
        tree2  = self.node_list_goal
        merged_tree = self.merged_node_list

        for i in range(self.max_iter):

            # ===== 1. 未找到路径前使用 Bi-RRT 采样 =====
            if not self.first_path_found:      
                rnd = self.sample_goal(20, tree2[0])
                rnd_node = Node(rnd[0], rnd[1], rnd[2])

                # ===== 2. 在当前树中扩展 =====
                nearest_ind = self.get_nearest_node_index(tree1, rnd)
                nearest_node = tree1[nearest_ind]

                steer_node = self.steer(nearest_node, rnd_node)
                new_node = self.apf_steer(steer_node, tree2[0])

                if not self.check_collision(new_node) and not self.check_edge_collision(nearest_node, new_node):
                    
                    # ===== 3. RRT*：choose parent =====
                    near_inds = self.find_near_nodes(new_node, tree1)
                    new_node = self.choose_parent(new_node, near_inds, tree1)

                    # ===== 4. RRT*：rewire =====
                    self.rewire(new_node, near_inds, tree1)

                    tree1.append(new_node)
                else:
                    continue

                # ===== 5. Bi-RRT：尝试连接另一棵树 =====
                connect_ind = self.get_nearest_node_index(tree2,
                                                        [new_node.x, new_node.y, new_node.z])
                connect_node = tree2[connect_ind]

                if not self.check_edge_collision(new_node, connect_node):

                    end_time = time.time()
                    # 判断 tree1 (new_node所在) 和 tree2 (connect_node所在) 谁是起点树
                    
                    if tree1[0] is self.start:
                        # 情况 A: tree1 是起点树，tree2 是终点树
                        # 此时 new_node 已经连在 Start 上了
                        # 我们需要把 connect_node (及其身后的父辈) 反转过来接在 new_node 上
                        self.revert_branch(connect_node, new_parent=new_node)
                        
                    else:
                        # 情况 B: tree1 是终点树，tree2 是起点树
                        # 此时 connect_node 已经连在 Start 上了
                        # 我们需要把 new_node (及其身后的父辈) 反转过来接在 connect_node 上
                        self.revert_branch(new_node, new_parent=connect_node)

                    # --- 此时，物理连接已经完成，从 Goal 到 Start 存在一条连续的 parent 链 ---

                    # 更新合并后的节点列表
                    merged_tree = tree1 + tree2
                    self.merged_node_list = merged_tree
                    
                    
                    # 生成路径（现在可以直接从 Goal 往回找了，不需要两段拼接！）
                    # 因为 revert_branch 已经把 Goal 的 parent 指向了 Start 方向
                    path_coords = self.generate_path_to_root(self.goal) 
                    
                    # 现在的 generate_path_to_root 出来的应该是 [Goal, ..., Start]
                    # 所以我们只需要反转它变成 [Start, ..., Goal]
                    first_path = path_coords[::-1]

                    self.goal.parent = None  # 重置 Goal 的 parent，防止影响后续优化搜索

                    
                    print(f"[首次成功] Bi-RRT* 找到路径，迭代 {i}")
                    print(f"[首次耗时] {end_time - start_time:.3f}s")
                    print(f"[首次路径长度] {self.calculate_path_length(first_path):.4f} 米")
                    self.first_path_found = True
                    self.c_best = self.calculate_path_length(first_path)
                    if not self.search_until_max_iter:
                        return first_path
                
                # ===== 6. 交换扩展方向 =====
                if tree1[0] is self.start:
                    self.node_list_start = tree1
                    self.node_list_goal = tree2
                else:
                    self.node_list_start = tree2
                    self.node_list_goal = tree1

                merged_tree = tree1 + tree2
                self.merged_node_list = merged_tree
                tree1, tree2 = tree2, tree1


            else:
                # 路径已找到后，使用单树 RRT* + Informed 采样继续优化
                rnd = self.informed_sample()
                # 如果采样失败（比如椭圆太小），直接跳过本次循环，防止卡死
                if rnd is None: 
                    continue
                rnd_node = Node(rnd[0], rnd[1], rnd[2])

                # 由于合并后所有节点都在 merged_node_list 中，我们对它进行扩展
                # 注意：为了保持一致性，建议统一把合并后的点都加到 node_list_start，或者直接用 merged_node_list
                current_tree = self.merged_node_list

                # 2. 找最近节点
                nearest_ind = self.get_nearest_node_index(current_tree, rnd)
                nearest_node = current_tree[nearest_ind]

                # 3. 扩展 (Steer)
                steer_node = self.steer(nearest_node, rnd_node)
                new_node = self.apf_steer(steer_node, self.goal) # 这里目标可以是 self.goal

                # 4. 碰撞检测
                if not self.check_collision(new_node) and not self.check_edge_collision(nearest_node, new_node):
                    
                    # 5. Choose Parent (标准 RRT* 步骤)
                    near_inds = self.find_near_nodes(new_node, current_tree)
                    new_node = self.choose_parent(new_node, near_inds, current_tree)
                    
                    # 6. Rewire (标准 RRT* 步骤 - 关键优化步骤)
                    self.rewire(new_node, near_inds, current_tree)

                    # 7. 加入树中
                    current_tree.append(new_node)

                # 如果找到了第一条可行路径，每次找到更优路径都更新 c_best
                temp_goal_ind = self.search_best_goal_node(self.merged_node_list)

                if temp_goal_ind is not None:
                    temp_path_coords = self.generate_final_path_from_node(temp_goal_ind, self.merged_node_list)
                    if  temp_path_coords:
                        # 补充终点
                        temp_path_coords.append([self.goal.x, self.goal.y, self.goal.z])
                        temp_path_len = self.calculate_path_length(temp_path_coords)
                        if temp_path_len < self.c_best:
                            self.c_best = self.merged_node_list[temp_goal_ind].cost + self.calc_distance(self.merged_node_list[temp_goal_ind], self.goal)
                            
        # 循环结束后，返回最优最终路径
        last_index = self.search_best_goal_node(self.merged_node_list)

        

        if last_index is not None and self.first_path_found:
            path_coords = self.generate_final_path_from_node(last_index, self.merged_node_list)
            if path_coords:
                # 补充终点
                path_coords.append([self.goal.x, self.goal.y, self.goal.z])
                last_time = time.time() - start_time
                print(f"[数据] 最终路径物理长度: {self.calculate_path_length(path_coords):.4f} 米")
                print(f"[数据] 最终路径计算时间: {last_time:.4f} 秒")
                return path_coords
            else:
                print("无法生成最终路径，后续迭代树出现环路。最终路径取初始路径长度")
                return first_path
    
        return None


# --------------------------sample-------------------------- 
    def sample_free(self):
        """
        随机采样一个点
        :return: 随机点的坐标 [x, y, z]
        """
        rnd_gen = random.Random()
        rnd = [rnd_gen.uniform(self.min_rand, self.max_rand),
               rnd_gen.uniform(self.min_rand, self.max_rand),
               rnd_gen.uniform(self.min_rand, self.z_rand)]
        return rnd
    
    def sample_goal(self, goal_sample_rate, target_node):
        """
        Bi-RRT 目标偏置采样
        :param goal_sample_rate: 0-100
        :param target_node: 对面树的根节点（Node）
        """
        if random.randint(0, 99) > goal_sample_rate:
            return self.sample_free()
        else:
            return [target_node.x, target_node.y, target_node.z]
    
    def informed_sample(self):
        # 非 informed 时回默认
        if (not self.use_informed_sampling) or self.c_best == float("inf"):
            return [
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.z_rand),
            ]

        # ------------------------------------------------------------------
        # 1. 计算椭球轴长
        # ------------------------------------------------------------------
        c_best = self.c_best
        c_min = self.c_min

        delta = max(c_best**2 - c_min**2, 0.0)
        a1 = c_best / 2.0
        a2 = math.sqrt(delta) / 2.0
        a3 = self.z_rand  # Z 方向不受限制

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

        return sample.tolist()
    
    
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

    def extend_star(self, tree_nodes, rnd):
        # 1. 最近节点
        nearest_ind = self.get_nearest_node_index(tree_nodes, rnd)
        nearest_node = tree_nodes[nearest_ind]

        # 2. steer / APF
        steer_node = self.steer(nearest_node, Node(rnd[0], rnd[1], rnd[2]))
        new_node = self.apf_steer(steer_node, self.goal)

        # 3. 碰撞检测
        if self.check_collision(new_node):
            return None
        if self.check_edge_collision(nearest_node, new_node):
            return None

        # 4. choose parent（局部最优）
        near_inds = self.find_near_nodes(new_node)
        new_node = self.choose_parent(new_node, near_inds) or new_node

        # 5. rewire（RRT* 精髓）
        self.rewire(new_node, near_inds)

        # 6. 加入当前树
        tree_nodes.append(new_node)

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

    def choose_parent(self, new_node, near_inds, tree):
        """
        选择最佳父节点，不steer直接连接
        :param new_node: 新节点
        :param near_inds: 附近节点的索引
        :return: 更新父节点后的新节点，而不是返回父节点
        
        """
        if not near_inds:
            return new_node

        costs = []
        for i in near_inds:
            near_node = tree[i]
            if  not self.check_edge_collision(near_node, new_node):
                costs.append(near_node.cost + self.calc_distance(near_node, new_node) + self.risk_cost(new_node))
            else:
                costs.append(float("inf"))  # 碰撞或无法连接

        min_cost = min(costs)
        if min_cost == float("inf"):
            return new_node

        min_ind = near_inds[costs.index(min_cost)]
        # new node不变化，直接连接
        new_node.cost = min_cost
        new_node.parent = tree[min_ind]
        return new_node     

    def rewire(self, new_node, near_inds, tree):
        

        for i in near_inds:
            near_node = tree[i]
            
            
            # 不生成新点，而是计算“如果直连”的距离
            dist_to_edge = self.calc_distance(new_node, near_node)

            # 计算新路径的 Cost
            new_cost = new_node.cost + dist_to_edge + self.risk_cost(near_node)

            # 只有 Cost 真的变小了，才做昂贵的碰撞检测
            if new_cost < near_node.cost:
                if not self.check_edge_collision(new_node, near_node):
                    # 只更新关系，不动坐标！
                    #print('check_rewire')

                    # ==========================================
                    # 【核心修复】 防止环路检测
                    # 如果 near_node 已经是 new_node 的祖先，
                    # 那么千万不能让 near_node 再认 new_node 做爹！
                    # ==========================================
                    # if self.is_ancestor(new_node, near_node):
                    #     # print("避免了一次死循环 rewire！")
                    #     continue

                    near_node.parent = new_node
                    near_node.cost = new_cost
                    self.propagate_cost_to_leaves(near_node, tree)

    def propagate_cost_to_leaves(self, parent_node, tree):
        '''
        递归更新子节点的成本
        '''
        #print('check_propagate')
        for node in tree:
            if node.parent == parent_node:
                node.cost = self.calc_distance(parent_node, node) + parent_node.cost + self.risk_cost(node)
                self.propagate_cost_to_leaves(node, tree)
        
        return

    def find_near_nodes(self, new_node, tree):
        """
        找到新节点附近的节点索引
        :param new_node: 新节点
        :return: 附近节点的索引列表
        """
        nnode = len(tree) + 1
        r = self.search_radius * math.sqrt(math.log(nnode) / nnode)  # 动态调整搜索半径
        r = min(r, self.search_radius) # 限制最大搜索半径
        r = max(r, 1.5*self.expand_dis) # 限制最小搜索半径
        dlist = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 + (node.z - new_node.z) ** 2 for node in tree]
        near_inds = [i for i in range(len(dlist)) if dlist[i] <= r ** 2]
        return near_inds
        
    def get_nearest_node_index(self, node_list, rnd):
        """
        找到树中距离随机点最近的节点的索引
        :param node_list: 当前树中的节点列表
        :param rnd: 随机采样点的坐标 [x, y]
        :return: 最近节点的索引
        """
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 + (node.z - rnd[2]) ** 2 for node in node_list]
        return dlist.index(min(dlist))
    
    def search_best_goal_node(self, tree, search_ratio=2):
        """
        在树中搜索除目标点本身成本最低（节点自身成本+离目标点直连距离最低）且可达的节点(需满足在search_ratio*expend_dis范围内且无碰撞)
        :return: 最优路径即最佳目标节点的索引，如果不存在则返回 None
        此函数只针对单颗树有效
        """
        best_index = None
        min_total_cost = float('inf')
        
        # 提前计算好搜索半径阈值
        search_radius = search_ratio * self.expand_dis

        # 使用 enumerate 一次遍历即可
        for i, node in enumerate(tree):
            # 1. 排除目标节点本身及其直接子节点
            if node is self.goal or node.parent is self.goal:
                continue
            
            # 2. 计算距离
            dist_to_goal = self.calc_dist_to_goal(node.x, node.y, node.z)
            
            # 3. 距离范围筛选
            if dist_to_goal > search_radius:
                continue
                
            # 4. 【重要优化】预估总成本剪枝
            # Total Cost = 当前节点路径代价 + 连接目标的欧式距离
            current_total_cost = node.cost + dist_to_goal
            
            # 如果当前计算出的成本已经不比已知最小成本小，
            # 就完全没必要做昂贵的碰撞检测了，直接跳过
            if current_total_cost >= min_total_cost:
                continue

            # 5. 碰撞检测 (只有当它是潜在的最优解时才检测)
            # 注意：这里假设 check_edge_collision 返回 True 代表有碰撞
            is_collision = self.check_edge_collision(node, self.goal)
            if not is_collision:
                min_total_cost = current_total_cost
                best_index = i

        return best_index


    def generate_final_path_from_node(self, end_node_index, tree):
        end_node = tree[end_node_index]
        path = [[end_node.x, end_node.y, end_node.z]]
        node = end_node
        
        # 安全计数器
        safe_count = 0 
        max_steps = len(tree) + 100 # 路径长度不可能超过节点总数
        
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y, node.z])
            
            # --- 死循环检测 ---
            safe_count += 1
            if safe_count > max_steps:
                print("【严重警告】检测到树中存在死循环环路！停止回溯。")
                return None
                
        return path[::-1]
    
    def generate_path_to_root(self, node):
        path = []
        while node is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        return path[::-1]
    
    def revert_branch(self, node, new_parent):
        """
        将终点树的一条分枝反转，使其嫁接到起点树上。
        :param node: 终点树中被连接的节点 (connect_node)
        :param new_parent: 起点树中的连接节点 (new_node)
        """
        # 1. 先回溯获取从当前节点到终点树根节点(Goal)的路径列表
        #    路径方向：connect_node -> parent -> ... -> Goal
        path_nodes = []
        curr = node
        while curr is not None:
            path_nodes.append(curr)
            curr = curr.parent

        # 2. 从连接点开始，依次反转父子关系
        #    现在的父节点变成了 new_parent (来自起点树)
        current_parent = new_parent
        
        for n in path_nodes:
            # 保存旧的父节点用于下一次迭代（因为 n.parent 即将被覆盖）
            # 注意：这里我们不需要旧的父节点来计算距离，因为欧氏距离是对称的
            # 但我们需要保留它来维持链条
            
            # 计算新成本：父节点成本 + 两点间距离
            dist = self.calc_distance(current_parent, n)
            n.cost = current_parent.cost + dist + self.risk_cost(n)
            
            # 【关键】反转指针：现在的 n 认 current_parent 为父
            n.parent = current_parent
            
            # 更新 current_parent 为当前节点，供列表下一个节点使用
            current_parent = n

        # 3. (可选) 成本传播
        # 因为反转了路径上的节点，原本挂在这些节点上的其他“子树”成本也变了
        # 这一步比较耗时，如果只为了求路径可以省略；如果为了全树一致性建议加上
        for n in path_nodes:
            self.propagate_cost_to_leaves(n, self.merged_node_list) # 注意这里列表可能混淆，需谨慎

    
    def is_ancestor(self, node, potential_ancestor):
        """
        检查 potential_ancestor 是否是 node 的祖先
        用于防止 Rewire 产生环路
        """
        curr = node
        # 往上回溯，看能不能碰到 potential_ancestor
        while curr is not None:
            if curr == potential_ancestor:
                return True
            curr = curr.parent
        return False

    # --------------------------工具函数--------------------------
    # def check_collision(self, node):
    #     """
    #     判断节点是否进入 crash 区（绝对禁止）。
    #     使用距离 <= (agent_R_crash + obs_R_crash) 的标准（XY 平面内，且 Z 重叠）。
    #     """
    #     for cx, cy, z_min, z_max, obs_R_crash, obs_R_risk in self.obstacle_list:
    #         if z_min <= node.z <= z_max:
    #             dist_xy = math.hypot(node.x - cx, node.y - cy)
    #             if dist_xy <= (self.R_crash + obs_R_crash):
    #                 return True
    #     return False
    
    def check_collision(self, node):
        """
        判断节点是否进入 crash 区。
        修正：加入 safety_margin，构建一个比视觉障碍物稍大的“隐形禁区”。
        """
        for cx, cy, z_min, z_max, obs_R_crash, obs_R_risk in self.obstacle_list:
            
            # 判定半径 = 机体 + 障碍物 + 安全余量
            collision_radius = self.R_crash + obs_R_crash + self.safety_margin
            
            # Z轴判定范围也扩大
            z_lower = z_min - self.R_crash - self.safety_margin
            z_upper = z_max + self.R_crash + self.safety_margin
            
            if z_lower <= node.z <= z_upper:
                dist_xy = math.hypot(node.x - cx, node.y - cy)
                if dist_xy <= collision_radius:
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
    
    def check_edge_collision(self, n1, n2, step_size=None):
        dist = self.calc_distance(n1, n2)
        if step_size is None:
            step_size = self.R_crash / 2.0
        if dist == 0: return False

        # ----------------------------------------
        # 1. 快速几何初筛 (加入 safety_margin)
        # ----------------------------------------
        for cx, cy, z_min, z_max, obs_R_crash, obs_R_risk in self.obstacle_list:
            
            # 计算包含安全余量的判定阈值
            collision_threshold = self.R_crash + obs_R_crash + self.safety_margin
            
            # Z轴安全范围
            obs_safe_z_min = z_min - self.R_crash - self.safety_margin
            obs_safe_z_max = z_max + self.R_crash + self.safety_margin
            
            path_z_min = min(n1.z, n2.z)
            path_z_max = max(n1.z, n2.z)
            
            # 如果高度完全错开，则绝对安全
            if path_z_max < obs_safe_z_min or path_z_min > obs_safe_z_max:
                continue

            # XY平面几何距离初筛
            dist_to_center = self.point_to_line_distance_xy(cx, cy, n1.x, n1.y, n2.x, n2.y)
            
            # 如果离散点还没查，几何距离就已经小于阈值了，这非常危险
            # 我们可以直接在这里做一次更加严格的判断，或者留给离散检测
            # 为了防止漏检，如果几何距离已经极其接近（例如小于阈值的 80%），可以直接判死刑
            if dist_to_center <= collision_threshold * 0.9: 
                 # 这是一个激进的优化：如果圆心到线段的垂直距离已经很小了，
                 # 且高度又有重叠，大概率是撞了。
                 # 为了严谨，我们还是交给下面的离散检测，但离散检测必须使用上面的 collision_threshold
                 pass

        # ----------------------------------------
        # 2. 离散检测 (复用 check_collision 即可)
        # ----------------------------------------
        # 因为 check_collision 已经加了 safety_margin，
        # 所以这里的采样点会自动遵循更严格的标准
        n_steps = int(dist / step_size) + 1
        dx = (n2.x - n1.x) / dist
        dy = (n2.y - n1.y) / dist
        dz = (n2.z - n1.z) / dist

        for i in range(n_steps):
            cx = n1.x + dx * step_size * i
            cy = n1.y + dy * step_size * i
            cz = n1.z + dz * step_size * i
            if self.check_collision(Node(cx, cy, cz)):
                return True
        
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



# ------------------测试用函数------------------
def check_point_validity(x, y, z, obstacle_list):
    """检查单个点是否在任何障碍物的 risk 范围内"""
    for cx, cy, z_min, z_max, R_crash, R_risk in obstacle_list:
        if z_min <= z <= z_max:
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist <= R_risk:  # 严格禁止进入 risk 区域
                return False
    return True

def generate_valid_tasks(num_tasks, env_map, seed=None):
    """
    生成 valid 的 (start, goal) 对
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    tasks = []
    obs_list = env_map["obstacles"]
    map_size = env_map["size"]
    z_size = env_map["z_size"]
    
    count = 0
    while len(tasks) < num_tasks:
        # 生成 Start
        sx = random.uniform(0, map_size)
        sy = random.uniform(0, map_size)
        sz = random.uniform(0, z_size) # 可以在空中
        
        # 生成 Goal
        gx = random.uniform(0, map_size)
        gy = random.uniform(0, map_size)
        gz = random.uniform(0, z_size)

        # 检查有效性
        if check_point_validity(sx, sy, sz, obs_list) and \
           check_point_validity(gx, gy, gz, obs_list):
            
            # 可选：确保起点终点不要太近 (例如 > 200m)
            if math.sqrt((sx-gx)**2 + (sy-gy)**2 + (sz-gz)**2) > 200:
                tasks.append(([sx, sy, sz], [gx, gy, gz]))
                count += 1
                # print(f"生成的第 {count} 个任务: Start -> Goal")
    
    return tasks



# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    
    # 1. 生成地图
    print("正在生成地图...")
    env_map = env_generator(
        rho=0.8, 
        map_size=1500,
        r_crash_range=(30, 50),
        r_risk_range=(3, 7),
        zmax_range=(30, 240),
        z_size=240,
        max_iter=5000,
        seed=39
    )
    obstacle_list = env_map["obstacles"]
    print(f"地图生成完毕，包含 {len(obstacle_list)} 个障碍物。")

    
    tasks = [([0,0,0],[1500.0, 1500.0, 50.0]),([0, 1500.0, 0],[1500.0, 0, 150.0])] #手动选择的起终点

    #tasks = generate_valid_tasks(5, env_map, seed=39)
    
    # 3. 运行测试
    success_times = []
    path_lengths = []
    
    # 设定 RRT* 参数 (Agent 自身尺寸设为 1.2m)
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    
    print(f"{'Task ID':<10} | {'Status':<10} | {'Time (s)':<10} | {'Length (m)':<10}")
    print("-" * 50)

    for i, (start, goal) in enumerate(tasks):
        # 初始化 RRT*
        planner = RRTStar(
            start=start, 
            goal=goal, 
            R_crash=r_agent_crash, 
            R_risk=r_agent_risk, 
            obstacle_list=obstacle_list, 
            rand_area=[0, env_map["size"], env_map["z_size"]], 
            expand_dis=30,    # 步长
            max_iter=3000,    # 迭代次数
            search_radius=150, # 搜索半径
            search_until_max_iter=True,  # 持续搜索以优化路径，保持常驻false，因为bi-rrt*利用剩余迭代次数优化不符合单RRT*的渐进最优性
            safety_margin=0.5   # 安全边距
        )
        start_time = time.time()
        path = planner.planning()
        
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        if path is not None:
            plen = planner.calculate_path_length(path)
            success_times.append(elapsed)
            path_lengths.append(plen)
            print(f"{i+1:<10} | {'Success':<10} | {elapsed:<10.4f} | {plen:<10.4f}")
            plot_tree_and_path(env_map, planner.merged_node_list, path)
            # # 优化路径
            # opt = PathOptimizer(env_map, safety_margin=0.5)

            # # 第一步：把折线拉直
            # path_pruned = opt.pruning_optimizer(path)

            # # 第二步：把直角磨圆（带自动防碰撞修正）
            # final_path_points = opt.smooth_optimizer(path_pruned)
            # plot_tree_and_path(env_map, planner.node_list, final_path_points)
            # optimal_length = calculate_path_length(final_path_points)
            # print(f"    优化后路径长度: {optimal_length:.4f} 米")
            

        else:
            print(f"{i+1:<10} | {'Failed':<10} | {elapsed:<10.4f} | {'N/A':<10}")

    # 4. 计算平均值
    print("-" * 50)
    if success_times:
        avg_time = sum(success_times) / len(success_times)
        avg_len = sum(path_lengths) / len(path_lengths)
        print(f"测试完成。")
        print(f"成功率: {len(success_times)/len(tasks)*100:.2f}% ")
        print(f"平均运行时间: {avg_time:.4f} 秒")
        print(f"平均路径长度: {avg_len:.4f} 米")
    else:
        print("所有任务均失败，请调整参数（如增加 max_iter 或减小 expand_dis）。")
