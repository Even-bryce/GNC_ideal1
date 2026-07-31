import numpy as np
import math
import time
from env_generator_for_data import env_generator, env_generator_maze, env_generator_cluster
from res_show import plot_tree_and_path

from planner_VRRT_star.planner_VRRT import VRRT
from planner_VRRT_star.planner_VRRT_star import VRRT_star
from planner_VRRT_star.planner_VRRT_star_APF import VRRT_star_APF

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

def run_planning(planner_type, 
                 start=(0.0, 0.0, 0.0), 
                 goal=(1500.0, 1500.0, 100.0)):
    """
    使用指定的 RRT 规划器，在指定种子生成的地图中进行一次规划，
    并返回简化后的航路点列表。

    参数：
        planner_type : str, 'VRRT', 'VRRT_star' 或 'VRRT_star_APF'
        seed         : int, 地图随机种子
        start, goal  : tuple, 起点和终点坐标

    返回：
        waypoints : list of [x,y,z] 或 None（规划失败时）
    """

    env_map = env_generator(
        rho=0.3,
        map_dim=(1500, 1500, 240),
        r_crash_range=(30, 50),
        r_risk_range=(3, 7),
        zmax_range=(30, 240),
        max_iter=10000,
        seed=2
    )
    # env_map = env_generator_cluster(
    #     map_dim=(1500, 1500, 240),
    #     num_clusters=33,
    #     chain_length_range=(2, 4), 
    #     r_center_range=(70, 120),
    #     r_edge_range=(20, 50), 
    #     r_risk_range=(10, 20),
    #     zmax_range=(240, 240),
    #     min_center_dist=200, 
    #     seed=7
    # )
    # env_map = env_generator_maze(
    #     grid_size=(4, 4),           # 4x4的网格，网格越多通道越窄越复杂
    #     map_dim=(1500, 1500, 240),
    #     r_crash_range=(40, 60),     # 为了给通道留出足够空间，半径相较于你原来设定的(80,125)稍微缩小了一些
    #     r_risk_range=(10, 20),
    #     zmax_range=(240, 240),
    #     seed=42
    # )
    obstacles = env_map["obstacles"]
    waypoints = [list(start), list(goal)]

    r_agent_crash = 1.2
    r_agent_risk = 1.7
    step_length = 50
    max_iter = 5000

    if planner_type == 'VRRT':
        rrt = VRRT(
            env_map=env_map,
            waypoints=waypoints,
            R_crash=r_agent_crash,
            R_risk=r_agent_risk,
            obstacle_list=obstacles,
            expand_dis=step_length,
            max_iter=max_iter
        )
    elif planner_type == 'VRRT_star':
        rrt = VRRT_star(
            env_map=env_map,
            waypoints=waypoints,
            R_crash=r_agent_crash,
            R_risk=r_agent_risk,
            obstacle_list=obstacles,
            expand_dis=step_length,
            search_radius=120,
            max_iter=max_iter,
            search_until_max_iter=True
        )
    elif planner_type == 'VRRT_star_APF':
        rrt = VRRT_star_APF(
            env_map=env_map,
            waypoints=waypoints,
            R_crash=r_agent_crash,
            R_risk=r_agent_risk,
            obstacle_list=obstacles,
            expand_dis=step_length,
            search_radius=120,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"未知的规划器类型: {planner_type}")

    result = rrt.planning()
    if result[0] is None:
        print("RRT 规划失败，未找到可行路径。")
        return None
    
    first_path_found, time_first, iter_find, path_len_list, _,  first_path, final_best_path = result
    plot_tree_and_path(env_map, first_path_found, final_best_path, waypoints)

    # 5. 路径简化（提取关键航路点）
    simplified_waypoints = find_straight_waypoint(final_best_path, env_map)
    if simplified_waypoints is not None:
        simplified_waypoints = [[round(x), round(y), round(z)] for x, y, z in simplified_waypoints]
    
    return simplified_waypoints

if __name__ == "__main__":
    wp = run_planning('VRRT_star')
    if wp:
        print(f"获得 {len(wp)} 个航路点：")
        print(wp)