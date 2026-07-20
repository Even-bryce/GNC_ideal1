import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from env_generator_for_data import env_generator, env_generator_maze, env_generator_cluster
from res_show import plot_map_and_waypoint, plot_tree_and_path

# 定义 Node 类
class Node:
    def __init__(self, x, y, z):
        self.x = x              # 节点的 x 坐标
        self.y = y              # 节点的 y 坐标
        self.z = z              # 节点的 z 坐标
        self.parent = None      # 节点的父节点，用于回溯路径
        self.cost = 0.0         # 从起点到该节点的路径成本

class RRT_star:
    def __init__(self, start, goal, R_crash, R_risk, obstacle_list, env_map, 
                 expand_dis=25, max_iter=1500, search_radius=20.0, 
                 search_until_max_iter=True):
        """
        初始化 RRT* 算法的参数
        :param start: 起点坐标 [x, y, z]
        :param goal: 目标坐标 [x, y, z]
        :param obstacle_list: 障碍物列表，每个障碍物为 [x, y, zmin, zmax, R_ob_crash, R_ob_risk]
        :param expand_dis: 树扩展的步长
        :param max_iter: 最大迭代次数
        :param search_radius: 搜索邻近节点的半径
        :param R_crash: 飞行器碰撞半径
        :param R_risk: 飞行器风险半径
        """
        self.start = Node(start[0], start[1], start[2])  # 创建起点节点
        self.goal = Node(goal[0], goal[1], goal[2])     # 创建目标节点
        self.env_map = env_map
        self.expand_dis = expand_dis           # 每次扩展的步长
        self.max_iter = max_iter               # 最大迭代次数
        self.obstacle_list = obstacle_list     # 存储障碍物列表
        self.node_list = [self.start]          # 树节点列表，初始化时只包含起点
        self.search_radius = search_radius     # 搜索邻近节点的半径
        self.R_crash = R_crash                 # 本体碰撞半径
        self.R_risk = R_risk                   # 本体风险半径
        self.search_until_max_iter = search_until_max_iter  # 是否持续搜索直到最大迭代次数

    def planning(self):
        """
        主规划函数，用于生成从起点到目标的路径
        返回首次找到的可行路径，和循环结束后找到的最优路径；否则返回 None
        """
        self.goal.cost = float('inf')
        self.goal.parent = None
        
        first_path_found = False
        time_first = None
        iteration_find_path = 0
        path_length_first = None
        path_length_final = np.inf
        first_path = None
        final_path = None

        start_time = time.time()
        
        for i in range(self.max_iter):  # 循环执行最大迭代次数
            # 随机采样
            rnd = self.sample_free()

            # 找到距离随机点最近的已有节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]
            
            # 计算扩展方向并生成新节点
            new_node = self.steer(nearest_node, Node(rnd[0], rnd[1], rnd[2]))
            
            # 检查新节点是否与障碍物碰撞
            if (not self.check_collision(new_node) and 
                    not self.check_edge_collision(nearest_node, new_node)):
                # 找到新节点附近的节点
                near_inds = self.find_near_nodes(new_node)
                # 选择最佳父节点
                new_node = self.choose_parent(new_node, nearest_node, near_inds)
                # 将新节点加入树中
                self.node_list.append(new_node)  
                # 重新连接邻近节点
                self.rewire(new_node, near_inds)

                # 未找到路径时
                if not first_path_found:
                    # 寻找可直接连接到目标点的节点
                    potential_goal_ind = self.search_best_goal_node()
                    # 若存在可直连的节点
                    if potential_goal_ind is not None:
                        first_path_found = True
                        # 生成路径
                        first_path = self.generate_final_path_from_node(potential_goal_ind)
                        # 补充终点
                        first_path.append([self.goal.x, self.goal.y, self.goal.z])
                        path_length_first = calculate_path_length(first_path)
                        
                        # 首次找到路径的迭代轮数与时间       
                        iteration_find_path = i
                        end_time = time.time()
                        time_first = end_time - start_time
                        
                        # 可设：找到首次路径就停止
                        if not self.search_until_max_iter:
                            return [first_path_found], [time_first], [iteration_find_path], [path_length_first], [path_length_first], first_path, first_path
        
        last_index = self.search_best_goal_node()

        if last_index is not None:
            # 用剩余迭代次数优化后的路径
            final_best_path = self.generate_final_path_from_node(last_index)
            # 补充终点
            final_best_path.append([self.goal.x, self.goal.y, self.goal.z])
            path_length_final = calculate_path_length(final_best_path)
            
            return [first_path_found], [time_first], [iteration_find_path], [path_length_first], [path_length_final], first_path, final_best_path

        return None, None, None, None, None, None, None
    
    def sample_free(self):
        """
        随机采样一个点
        :return: 随机点的坐标 [x, y, z]
        """
        max_x_rand = self.env_map["map_dim"][0]
        max_y_rand = self.env_map["map_dim"][1]
        z_rand = self.env_map["map_dim"][2]
        rnd_gen = random.Random()
        rnd = [rnd_gen.uniform(0, max_x_rand),
               rnd_gen.uniform(0, max_y_rand),
               rnd_gen.uniform(0, z_rand)]
        return rnd
    
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
    
    def get_nearest_node_index(self, node_list, rnd):
        """
        找到树中距离随机点最近的节点的索引
        :param node_list: 当前树中的节点列表
        :param rnd: 随机采样点的坐标 [x, y]
        :return: 最近节点的索引
        """
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 + (node.z - rnd[2]) ** 2 for node in node_list]
        return dlist.index(min(dlist))

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
        # 限制最大、最小搜索半径
        r = min(r, self.search_radius)
        r = max(r, 2 * self.expand_dis)
        dlist = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 + (node.z - new_node.z) ** 2 for node in self.node_list]
        near_inds = [i for i in range(len(dlist)) if dlist[i] <= r ** 2]
        return near_inds

    def choose_parent(self, new_node, nearest_node, near_inds):
        """
        选择最佳父节点
        :param new_node: 新节点
        :param near_inds: 附近节点的索引
        :return: 更新后的新节点
        """
        # 以nearest_node作为初始父节点
        min_node = nearest_node
        min_cost = nearest_node.cost + self.calc_distance(nearest_node, new_node)

        for i in near_inds:
            near_node = self.node_list[i]
            # 计算“起点——临近节点——新节点”的路径成本
            new_cost = near_node.cost + self.calc_distance(new_node, near_node) + self.risk_cost(new_node)
            
            # 路径成本减少，则进行碰撞检测
            if new_cost < min_cost:
                if not self.check_edge_collision(new_node, near_node):
                    min_node = near_node
                    min_cost = new_cost

        new_node.cost = min_cost
        new_node.parent = min_node
        return new_node
    
    def propagate_cost_to_leaves(self, parent_node):
        '''
        递归更新子节点的成本
        '''
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_distance(parent_node, node) + parent_node.cost + self.risk_cost(node)
                self.propagate_cost_to_leaves(node)

    def steer(self, from_node, to_node):
        """
        从 from_node 向 to_node 扩展一个新节点
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

    def rewire(self, new_node, near_inds):
        """
        重新连接邻近节点以优化路径
        :param new_node: 新节点
        :param near_inds: 附近节点的索引
        """
        for i in near_inds:
            near_node = self.node_list[i]
            
            # 计算“起点——新节点——临近节点”的路径成本
            dist_to_edge = self.calc_distance(new_node, near_node)
            new_cost = new_node.cost + dist_to_edge + self.risk_cost(near_node)

            # 路径成本减少，则进行碰撞检测
            if new_cost < near_node.cost:
                if not self.check_edge_collision(new_node, near_node):
                    # 更新节点关系
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    self.propagate_cost_to_leaves(near_node)
                    
    def search_best_goal_node(self):
        """
        在树中搜索距目标点一定半径内且可达的节点
        :return: 最佳目标节点的索引，如果不存在则返回 None
        """
        best_index = None
        min_total_cost = float('inf')
        
        # 避免遍历过多节点，仅在终点周围搜索
        search_radius = 5 * self.expand_dis
        
        for i, node in enumerate(self.node_list):
            # 计算树中所有节点到目标的距离
            dist_to_goal = self.calc_distance(node, self.goal)
            
            # 如果节点在搜索半径内
            if dist_to_goal <= search_radius:
                # 计算“起点——节点——直连终点”的路径成本
                total_cost = node.cost + dist_to_goal
                # 判断节点到终点是否无碰撞
                if not self.check_edge_collision(node, self.goal):
                    if total_cost < min_total_cost:
                        min_total_cost = total_cost
                        best_index = i
        
        return best_index
    
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
                    penalty += 0.0 / margin  # 权重可调
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
    # 1. 生成地图
    print("正在生成地图...")
    # env_map = env_generator(
    #     rho=0.4, 
    #     map_dim=(1500, 1500, 240),
    #     r_crash_range=(30, 50),
    #     r_risk_range=(3, 7),
    #     zmax_range=(30, 240),
    #     max_iter=10000,
    #     seed=40
    # )
    env_map = env_generator_cluster(
        map_dim=(1500, 1500, 240), 
        num_clusters=20,
        chain_length_range=(1, 4), 
        r_center_range=(100, 150),
        r_edge_range=(20, 50), 
        r_risk_range=(10, 20),
        zmax_range=(240, 240), 
        min_center_dist=200, 
        seed=7)
    obstacle_list = env_map["obstacles"]
    print(f"地图生成完毕，包含 {len(obstacle_list)} 个障碍物。")
    
    tasks = [([0, 0, 0], [1500, 1500, 100])]
    # plot_map_and_waypoint(env_map, ([0, 0, 0], [1500, 1500, 240]))

    # 3. 运行测试
    success_times = []
    path_lengths = []
    
    # 设定 RRT* 参数
    r_agent_crash = 1.2
    r_agent_risk = 1.7
    env_results = []
    print(f"{'Task':<4} | {'Status':<7} | {'Iter':<6} | {'Time_first':<6}  | {'Length_first':<6} | {'Time_final':<6} | {'Length_final':<6}")
    print("-" * 80)
    
    # 对每个起终点，进行num_pf_tests次规划
    num_of_tests = 10
    for i, (start, goal) in enumerate(tasks):
        env_first_times = []
        env_final_times = []
        env_iters = []
        env_first_lengths = []
        env_final_lengths = []
        env_success_count = 0
        for j in range(num_of_tests):
            # 初始化 RRT*
            rrt_star = RRT_star(
                start=start, 
                goal=goal, 
                R_crash=r_agent_crash, 
                R_risk=r_agent_risk, 
                obstacle_list=obstacle_list, 
                env_map=env_map,  
                expand_dis=20,
                max_iter=2000,
                search_radius=100
            )
            start_time = time.time()
            result = rrt_star.planning()
            end_time = time.time()
            
            time_final = end_time - start_time
            
            if result[0] is None:
                time_first = result[1]
                iteration_find_path = result[2]
            else:
                time_first = result[1][0]
                iteration_find_path = result[2][0]
            first_path = result[5]
            final_best_path = result[6]
            
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
            
    # 4. 计算平均值
    print("=" * 80)
    print(f"{'Task':<4} | {'Rate(%)':<4} | {'Iter':<6} | {'Time_first':<6}  | {'Length_first':<6} | {'Time_final':<6} | {'Length_final':<6}")
    print("-" * 80)

    for res in env_results:
        print(f"{res['env_id']:<4} | {res['success_rate']:<7.1f} | {res['avg_iters']:<6.1f} | {res['avg_time_first']:<11.4f} | {res['avg_length_first']:<12.1f} | {res['avg_time_final']:<10.4f} | {res['avg_length_final']:<10.1f}")

    print("-" * 80)
