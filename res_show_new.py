import numpy as np
import pyvista as pv

def plot_map_pv(env_map):
    """
    使用 PyVista 绘制地图 (替代原来的 plot_map)
    只显示环境障碍物，用于快速检查地图生成情况
    """
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')
    
    _add_obstacles_to_plotter(plotter, env_map)
    
    # 添加地图边界框（新 API）
    map_size = env_map["size"]
    z_size = env_map["z_size"]
    bounds = [0, map_size, 0, map_size, 0, z_size]

    # ✔ PyVista 正确使用 show_grid，而不是 add_bounds_axes
    plotter.add_axes()  # 这条命令画出三条从原点出发的坐标轴
    plotter.show_grid(bounds=bounds, color='black')
    
    print("地图预览中... 关闭窗口以继续。")
    plotter.show()

def plot_tree_and_path_pv(env_map, node_list=None, path=None):
    """
    使用 PyVista 绘制 RRT 树和最终路径 (修复版)
    修复内容：
    1. 显式指定 lines 参数，解决 Empty mesh 报错
    2. 强制转换 float 类型，消除 UserWarning
    """
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')
    
    # 1. 绘制障碍物
    _add_obstacles_to_plotter(plotter, env_map)

    # 2. 绘制 RRT 树
    if node_list is not None and len(node_list) > 1:
        print(f"正在构建 RRT 树 ({len(node_list)} 个节点)...")
        
        line_segments = []
        for node in node_list:
            if node.parent:
                p_start = [node.x, node.y, node.z]
                p_end = [node.parent.x, node.parent.y, node.parent.z]
                line_segments.append([p_start, p_end])
        
        if line_segments:
            # === 修复 1: 强制转换为 float 类型，消除警告 ===
            line_segments = np.array(line_segments).astype(float)
            
            # 构造 PolyData
            n_lines = len(line_segments)
            flat_points = line_segments.reshape(-1, 3)
            
            # 构造连接矩阵 [2, id1, id2, 2, id3, id4, ...]
            # 2 表示每个单元有2个点（即线段）
            padding = np.full((n_lines, 1), 2, dtype=int)
            indices = np.arange(2 * n_lines).reshape(n_lines, 2)
            cells = np.hstack((padding, indices)).flatten()
            
            # === 修复 2: 关键修改！必须使用 lines=cells ===
            # 如果不写 lines=，PyVista 会以为这些是面(faces)，导致 tube 生成失败
            tree_mesh = pv.PolyData(flat_points, lines=cells)
            
            # 管线化
            tree_tubes = tree_mesh.tube(radius=0.2) # 半径可以根据需要调整
            
            plotter.add_mesh(tree_tubes, color='lime', opacity=0.5, label='RRT Tree')

    # 3. 绘制最终路径
    if path is not None:
        path_arr = np.array(path).astype(float) # 同样转为 float
        if len(path_arr) > 1:
            spline = pv.lines_from_points(path_arr)
            tube = spline.tube(radius=0.8) 
            plotter.add_mesh(tube, color='red', label='Final Path')
            
            plotter.add_mesh(pv.Sphere(radius=2, center=path_arr[0]), color='blue', label='Start')
            plotter.add_mesh(pv.Sphere(radius=2, center=path_arr[-1]), color='orange', label='Goal')

    plotter.add_axes()
    # -----------------------------------------------------------
    map_size = env_map["size"]      # 例如 1500
    z_size = env_map["z_size"]      # 例如 300
    
    # 定义边界：[x_min, x_max, y_min, y_max, z_min, z_max]
    bounds = [0, map_size, 0, map_size, 0, z_size]

    # show_grid 负责显示刻度尺
    # bounds 参数强制规定了刻度尺的范围，这样即使你的路径很短，
    # 视角也会被拉大到整个 1500x1500 的地图范围
    plotter.show_grid(bounds=bounds, color='black')
    plotter.add_legend()
    
    print("3D 窗口已打开。")
    plotter.show()

def _add_obstacles_to_plotter(plotter, env_map):
    """
    辅助函数：将圆柱体添加到场景中
    """
    obstacles = env_map["obstacles"]
    
    for obs in obstacles:
        xc, yc, zmin, zmax, r_crash, r_risk = obs
        
        height = zmax - zmin
        if height <= 0: continue
        z_center = zmin + height / 2.0
        
        # 1. 绘制 Crash 区域 (实心障碍物)
        # 关键：Opacity=0.6，这样你能透过它看到内部的路径！
        cyl_crash = pv.Cylinder(center=(xc, yc, z_center), direction=(0, 0, 1), 
                                radius=r_crash, height=height, resolution=40)
        plotter.add_mesh(cyl_crash, color='#555555', opacity=0.6) 
        
        # 2. 绘制 Risk 区域 (仅显示线框或极淡的颜色)
        # 用线框模式显示风险区，干扰更小
        cyl_risk = pv.Cylinder(center=(xc, yc, z_center), direction=(0, 0, 1), 
                               radius=r_risk, height=height, resolution=40)
        plotter.add_mesh(cyl_risk, color='orange', style='wireframe', opacity=0.2)

# ==========================================
# 兼容性接口：如果你不想改主程序的调用代码
# 可以把原来的函数名指向新的 PyVista 版本
# ==========================================
plot_map = plot_map_pv
plot_tree_and_path = plot_tree_and_path_pv

if __name__ == "__main__":
    # 测试代码
    # 模拟一个简单的 Node 类
    class Node:
        def __init__(self, x, y, z, parent=None):
            self.x, self.y, self.z = x, y, z
            self.parent = parent

    # 1. 生成测试数据
    mock_env = {
        "size": 100, "z_size": 100,
        "obstacles": [
            (50, 50, 0, 80, 10, 15), # 中间的柱子
            (20, 80, 0, 50, 5, 10)
        ]
    }
    
    # 模拟树
    n1 = Node(10, 10, 10)
    n2 = Node(30, 30, 30, n1)
    n3 = Node(45, 45, 40, n2) # 撞向柱子
    nodes = [n1, n2, n3]
    
    # 模拟路径 (穿过柱子演示效果)
    path = [[10, 10, 10], [30, 30, 30], [45, 45, 40], [60, 60, 60]]

    # 2. 调用绘图
    # plot_map_pv(mock_env)
    plot_tree_and_path_pv(mock_env, nodes, path)