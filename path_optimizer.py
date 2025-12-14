import numpy as np
import math
from scipy.interpolate import splprep, splev

class PathOptimizer:
    def __init__(self, env_map, safety_margin=5):
        """
        初始化优化器
        :param env_map: 你的 env_generator 生成的字典
        :param safety_margin: 碰撞检测的安全余量 (即 r_crash + safety_margin)
        """
        self.obstacles = env_map['obstacles']
        self.margin = safety_margin
        
        # 提取障碍物数据以加速计算 (N, 6) -> x, y, zmin, zmax, r_crash, r_risk
        if len(self.obstacles) > 0:
            self.obs_array = np.array(self.obstacles)
        else:
            self.obs_array = np.empty((0, 6))

    def _dist_segment_to_point_2d(self, p1, p2, point):
        """
        计算线段 p1-p2 到点 point 的最短 2D 距离 (忽略 z 轴)
        p1, p2: [x, y, z]
        point: [x, y]
        """
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        point_2d = point
        
        # 向量运算
        line_vec = p2_2d - p1_2d
        p1_to_point = point_2d - p1_2d
        
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq == 0:
            return np.linalg.norm(p1_to_point)
            
        # 投影参数 t
        t = np.dot(p1_to_point, line_vec) / line_len_sq
        
        # 限制 t 在 [0, 1] 之间，即限制在线段范围内
        t = np.clip(t, 0, 1)
        
        # 找到线段上距离圆心最近的点
        closest_point = p1_2d + t * line_vec
        
        return np.linalg.norm(point_2d - closest_point)

    def is_segment_collision_free(self, p1, p2):
        """
        检测线段 p1->p2 是否无碰撞
        """
        # 1. 快速包围盒检测 (AABB)
        # 如果线段的 z 范围完全在障碍物 z 范围之外，直接排除
        seg_z_min = min(p1[2], p2[2])
        seg_z_max = max(p1[2], p2[2])
        
        for obs in self.obs_array:
            ox, oy, oz_min, oz_max, r_crash, _ = obs
            
            # Z轴高度检查：如果线段完全在圆柱下方或完全在圆柱上方，则安全
            if seg_z_max < oz_min or seg_z_min > oz_max:
                continue
            
            # 2D 平面距离检查
            dist = self._dist_segment_to_point_2d(p1, p2, np.array([ox, oy]))
            
            if dist < (r_crash + self.margin):
                return False # 发生碰撞
                
        return True

    def is_path_collision_free(self, path):
        """
        检测整条离散路径是否安全 (用于检测 B 样条生成的密集点)
        这里为了效率，检测每个离散点是否在障碍物内
        """
        path = np.array(path)
        if len(path) == 0: return True
        
        # 批量检测点
        xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]
        
        for obs in self.obs_array:
            ox, oy, oz_min, oz_max, r_crash, _ = obs
            safe_r_sq = (r_crash + self.margin) ** 2
            
            # 筛选出 Z 轴范围内有风险的点
            # 逻辑：(z >= zmin) AND (z <= zmax)
            z_mask = (zs >= oz_min) & (zs <= oz_max)
            
            if not np.any(z_mask):
                continue
                
            # 对这些风险点计算 XY 平面距离平方
            dx = xs[z_mask] - ox
            dy = ys[z_mask] - oy
            d_sq = dx*dx + dy*dy
            
            if np.any(d_sq < safe_r_sq):
                return False # 碰撞
                
        return True

    def pruning_optimizer(self, path, max_iter=100):
        """
        第一阶段：剪枝 (Shortcut)
        """
        path = np.array(path)
        if len(path) < 3: return path
        
        # 简单的贪婪剪枝
        # 每次尝试把当前点连接到尽可能远的后续点
        new_path = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # 从终点倒着往前找
            found_shortcut = False
            for target_idx in range(len(path) - 1, current_idx + 1, -1):
                # 如果是相邻点，直接连，不需要检测
                if target_idx == current_idx + 1:
                    new_path.append(path[target_idx])
                    current_idx = target_idx
                    found_shortcut = True
                    break
                
                # 检测长线段是否碰撞
                if self.is_segment_collision_free(path[current_idx], path[target_idx]):
                    new_path.append(path[target_idx])
                    current_idx = target_idx
                    found_shortcut = True
                    break
            
            if not found_shortcut:
                # 理论上不会进这里，因为 target_idx 循环最后会退化到 current_idx + 1
                current_idx += 1
                
        return np.array(new_path)

    def smooth_optimizer(self, path, sub_division_depth=0, max_depth=3):
        """
        第二阶段：递归 B 样条平滑
        :param path: 关键控制点 (通常是剪枝后的路径)
        :param sub_division_depth: 当前递归深度
        :param max_depth: 最大细分深度，防止死循环
        """
        path = np.array(path)
        if len(path) < 3: return path # 点太少无法生成 B 样条
        
        # 1. 数据预处理：去重 (B样条对重复点很敏感)
        # 简单的去重逻辑：如果两点距离极近，丢弃后一个
        unique_path = [path[0]]
        for i in range(1, len(path)):
            if np.linalg.norm(path[i] - unique_path[-1]) > 1e-3:
                unique_path.append(path[i])
        unique_path = np.array(unique_path)
        
        if len(unique_path) < 3: return unique_path

        # 2. 尝试生成 B 样条
        try:
            # k=3 (三次样条), s=平滑因子 (可以根据需要调整，s=0 表示穿过所有点)
            # 这里的 s 可以设小一点，或者设为 None 让算法自己估计
            tck, u = splprep(unique_path.T, k=3, s=0.5) 
            
            # 生成密集点用于检测
            u_new = np.linspace(0, 1, num=len(unique_path)*10)
            smooth_path_points = np.array(splev(u_new, tck)).T
            
            # 3. 碰撞检测
            if self.is_path_collision_free(smooth_path_points):
                # 成功！路径平滑且无碰撞
                return smooth_path_points
            
        except Exception as e:
            print(f"Spline generation failed: {e}, fallback to linear.")
            # 如果样条生成失败（比如点共线等极端情况），回退到细分
            pass

        # 4. 如果碰撞或失败，且未达到最大深度 -> 细分 (Subdivision)
        if sub_division_depth < max_depth:
            # print(f"Smoothing collision detected, subdividing... (Depth {sub_division_depth})")
            new_control_points = []
            for i in range(len(unique_path) - 1):
                p1 = unique_path[i]
                p2 = unique_path[i+1]
                mid_point = (p1 + p2) / 2.0
                new_control_points.append(p1)
                new_control_points.append(mid_point)
            new_control_points.append(unique_path[-1])
            
            return self.smooth_optimizer(new_control_points, sub_division_depth + 1, max_depth)
        
        else:
            # 5. 达到最大深度依然撞，只能退化回原始折线 (保证可行性)
            # print("Max recursion depth reached, returning linear path.")
            # 为了保持返回格式一致，我们在折线上插值一些点
            dense_linear_path = []
            for i in range(len(unique_path)-1):
                p_start = unique_path[i]
                p_end = unique_path[i+1]
                steps = np.linspace(0, 1, 10)
                for s in steps:
                    dense_linear_path.append(p_start + (p_end - p_start) * s)
            dense_linear_path.append(unique_path[-1])
            return np.array(dense_linear_path)

# ==========================================
# 使用示例 (配合你的 Map 代码)
# ==========================================
if __name__ == "__main__":
    
    # 1. 生成地图
    # 注意：这里需要你上面提供的 env_generator 函数
    # from map_module import env_generator (如果你分文件放)
    
    # --- 这里为了能跑通，我手动模拟一个简单的 env_map ---
    # 实际运行时请使用你的 env_generator(seed=42)
    env_map = {
        'obstacles': [
            (50, 50, 0, 100, 10, 15), # x, y, zmin, zmax, r_crash, r_risk
            (80, 80, 0, 100, 10, 15)
        ]
    }
    
    # 2. 模拟一条粗糙的 RRT 路径 (绕过障碍物)
    # 假设起点 (0,0,0) 终点 (100,100,50)
    raw_path = np.array([
        [0, 0, 0],
        [20, 10, 5],
        [40, 60, 20], # 绕远了
        [45, 55, 25], # 抖动
        [60, 40, 30], # 绕回来
        [90, 90, 40],
        [100, 100, 50]
    ])
    
    # 3. 初始化优化器
    optimizer = PathOptimizer(env_map, safety_margin=2.0)
    
    # 4. 执行剪枝
    pruned_path = optimizer.pruning_optimizer(raw_path)
    print(f"Original nodes: {len(raw_path)} -> Pruned nodes: {len(pruned_path)}")
    
    # 5. 执行平滑
    smoothed_path = optimizer.smooth_optimizer(pruned_path)
    print(f"Smoothed path points: {len(smoothed_path)}")
    
    # 6. 画图对比 (需要 matplotlib)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画障碍物 (简易画法)
    for obs in env_map['obstacles']:
        ox, oy, zmin, zmax, r, _ = obs
        # 画圆柱上下底面圆周
        theta = np.linspace(0, 2*np.pi, 20)
        x_cir = ox + r * np.cos(theta)
        y_cir = oy + r * np.sin(theta)
        ax.plot(x_cir, y_cir, zmin, 'r-', alpha=0.3)
        ax.plot(x_cir, y_cir, zmax, 'r-', alpha=0.3)
        # 画几条竖线
        for i in range(0, 20, 5):
            ax.plot([x_cir[i], x_cir[i]], [y_cir[i], y_cir[i]], [zmin, zmax], 'r-', alpha=0.3)

    # 画路径
    ax.plot(raw_path[:,0], raw_path[:,1], raw_path[:,2], 'k--', label='Raw RRT', alpha=0.4)
    ax.plot(pruned_path[:,0], pruned_path[:,1], pruned_path[:,2], 'bo-', label='Pruned', linewidth=1)
    ax.plot(smoothed_path[:,0], smoothed_path[:,1], smoothed_path[:,2], 'g-', label='Smoothed', linewidth=2.5)
    
    ax.legend()
    ax.set_title("Path Optimization: Pruning + Recursive B-Spline")
    plt.show()