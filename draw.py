import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 新增绘图函数
def plot_algorithm_comparison_across_maps(avg_matrices, map_names=None, algo_names=None, save_dir=None):
    """
    绘制三种地图下各算法的指标折线对比图（8个子图）

    Parameters:
    -----------
    avg_matrices : list of lists
        包含三个矩阵的列表，每个矩阵为二维列表（行=算法，列=指标）
        指标顺序固定：[成功率(%), 首次时间(s), 首次长度, 首次平滑度(deg),
                      迭代次数, 最终时间(s), 最终长度, 最终平滑度(deg)]
        三个矩阵的行顺序必须一致（同一行对应同一算法）
    map_names : list of str, optional
        三种地图的名称，默认 ['cluster', 'clutter', 'maze']
    algo_names : list of str, optional
        算法名称列表，长度等于矩阵行数。若未提供，则使用 'Algo1', 'Algo2', ...
    save_dir : str, optional
        图片保存目录，若为 None 则不保存，仅显示
    """
    # 默认地图名称
    if map_names is None:
        map_names = ['cluster', 'clutter', 'maze']
    if len(map_names) != 3:
        raise ValueError("必须提供恰好三种地图的名称")

    if len(avg_matrices) != 3:
        raise ValueError("必须提供恰好三个平均值矩阵（对应三种地图）")

    # 转换为 numpy 数组
    mats = [np.array(mat, dtype=float) for mat in avg_matrices]
    n_algo = mats[0].shape[0]
    for m in mats:
        if m.shape[0] != n_algo:
            raise ValueError("所有矩阵的行数（算法数）必须相同")
        if m.shape[1] != 8:
            raise ValueError("每个矩阵必须包含 8 列指标")

    # 算法名称
    if algo_names is None:
        algo_names = [f'Algo{i+1}' for i in range(n_algo)]
    else:
        if len(algo_names) != n_algo:
            raise ValueError("算法名称列表长度必须与矩阵行数一致")

    # 指标名称（8个）
    metric_names = [
        'Success rate (%)',
        'Time First (s)',
        'Length First',
        'Smoothness First (deg)',
        'Iterations Total',
        'Time Final (s)',
        'Length Final',
        'Smoothness Final (deg)'
    ]

    # 准备绘图数据：每个指标一个列表，每个列表包含三个地图的向量
    all_data = []
    for metric_idx in range(8):
        map_vectors = []
        for map_idx in range(3):
            vec = mats[map_idx][:, metric_idx].copy()
            # 如果是迭代次数（索引4），将 A*（第一个算法）的值设为 NaN
            if metric_idx == 4:
                vec[0] = np.nan
            map_vectors.append(vec)
        all_data.append(map_vectors)  # shape: (8, 3, n_algo)

    # 创建 2x4 子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Algorithm Comparison Across Maps (Cluster / Clutter / Maze)', fontsize=16)

    # 三种地图的样式
    styles = [
        {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        {'color': 'green', 'marker': 's', 'linestyle': '-'},
        {'color': 'red', 'marker': '^', 'linestyle': '-'}
    ]

    # 绘制每个子图
    for idx, metric_name in enumerate(metric_names):
        row = idx // 4
        col = idx % 4
        ax = axes[row][col]

        map_vectors = all_data[idx]

        for map_i, vec in enumerate(map_vectors):
            x = np.arange(n_algo)
            mask = ~np.isnan(vec)
            ax.plot(x[mask], vec[mask],
                    label=map_names[map_i],
                    color=styles[map_i]['color'],
                    marker=styles[map_i]['marker'],
                    linestyle=styles[map_i]['linestyle'],
                    linewidth=2, markersize=8)

        # 横轴使用简短编号（1, 2, 3, ...）
        ax.set_xticks(np.arange(n_algo))
        ax.set_xticklabels([str(i+1) for i in range(n_algo)])
        ax.set_title(metric_name)
        ax.grid(True, linestyle='--', alpha=0.6)

        # 纵轴范围微调
        if metric_name == 'Success rate (%)':
            ax.set_ylim(0, 105)
        else:
            all_vals = np.concatenate([v[~np.isnan(v)] for v in map_vectors])
            if len(all_vals) > 0:
                min_val = np.min(all_vals)
                max_val = np.max(all_vals)
                margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.5
                ax.set_ylim(min_val - margin, max_val + margin)

    # ----- 添加图例（两个）-----
    # 1. 地图图例（右上角）
    map_handles = []
    for map_i, name in enumerate(map_names):
        line = plt.Line2D([], [],
                          color=styles[map_i]['color'],
                          marker=styles[map_i]['marker'],
                          linestyle=styles[map_i]['linestyle'],
                          linewidth=2, markersize=8,
                          label=name)
        map_handles.append(line)
    fig.legend(handles=map_handles, loc='upper right', bbox_to_anchor=(0.98, 0.96),
               ncol=1, fontsize=10, title='Map Type')

    # 2. 算法图例（底部，横向排列）
    # 使用灰色方块作为标识（避免与地图颜色混淆）
    algo_handles = [Patch(facecolor='gray', edgecolor='gray', label=name) for name in algo_names]
    fig.legend(handles=algo_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(algo_names), fontsize=9, title='Algorithms')

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])  # 为底部图例留空间
    plt.subplots_adjust(bottom=0.12)          # 进一步调整底部空间

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'algorithm_comparison_across_maps.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()
    
def plot_planner_comparison(planner_names, success_rates, avg_data, err_data,
                            step, map_type, save_dir=None):
    """
    绘制规划器对比图（合并为一张2x4子图）
    """
    colors = ["#ff0000", "#ff7700", "#ffee00", "#183b00ec", "#51ff00", "#00eeff",
              "#0037ff", "#4c00ff", "#bb00ff", "#fb00ff", "#ff0066", "#4b000253"]
    if len(planner_names) > len(colors):
        colors = colors * (len(planner_names) // len(colors) + 1)

    legend_elements = [Patch(facecolor=colors[i], edgecolor='black',
                             label=planner_names[i])
                       for i in range(len(planner_names))]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # 子图数据顺序：行1 (0,0)~(0,3), 行2 (1,0)~(1,3)
    subplot_data = [
        (0, 0, success_rates, None, 'Success rate (%)'),
        (0, 1, avg_data['max_time_first'], err_data['max_time_first'], 'Time First (s)'),
        (0, 2, avg_data['len_first'], err_data['len_first'], 'Length First'),
        (0, 3, avg_data['smooth_first'], err_data['smooth_first'], 'Smoothness First (deg)'),
        (1, 0, avg_data['sum_iter'], err_data['sum_iter'], 'Iterations Total'),
        (1, 1, avg_data['time_final'], err_data['time_final'], 'Time Final (s)'),
        (1, 2, avg_data['len_final'], err_data['len_final'], 'Length Final'),
        (1, 3, avg_data['smooth_final'], err_data['smooth_final'], 'Smoothness Final (deg)')
    ]

    for row, col, y_vals, y_err, title in subplot_data:
        ax = axes[row][col]
        x_pos = np.arange(len(planner_names))
        for i, (val, err) in enumerate(zip(y_vals, y_err if y_err is not None else [None]*len(y_vals))):
            ax.bar(x_pos[i], val, yerr=err, color=colors[i], capsize=5,
                   edgecolor='black', width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([''] * len(x_pos))
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 纵轴范围优化
        if title == 'Success rate (%)':
            ax.set_ylim(0, 110)
        elif 'Iterations' not in title:
            data_min = min(y_vals) if y_vals else 0
            if data_min > 0:
                ax.set_ylim(bottom=data_min * 0.9)

        # ---- 在最终长度和平滑度上标注优化百分比 ----
        if title == 'Length Final':
            y_min, y_max = ax.get_ylim()
            offset = (y_max - y_min) * 0.02
            for i, val_final in enumerate(y_vals):
                val_first = avg_data['len_first'][i]
                if val_first > 0:
                    reduction = (val_first - val_final) / val_first * 100
                    label = f'-{reduction:.1f}%' if reduction >= 0 else f'+{-reduction:.1f}%'
                else:
                    label = 'N/A'
                ax.text(x_pos[i], val_final + offset, label,
                        ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

        if title == 'Smoothness Final (deg)':
            y_min, y_max = ax.get_ylim()
            offset = (y_max - y_min) * 0.02
            for i, val_final in enumerate(y_vals):
                val_first = avg_data['smooth_first'][i]
                if val_first > 0:
                    reduction = (val_first - val_final) / val_first * 100
                    label = f'-{reduction:.1f}%' if reduction >= 0 else f'+{-reduction:.1f}%'
                else:
                    label = 'N/A'
                ax.text(x_pos[i], val_final + offset, label,
                        ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.98, 0.96), ncol=1, fontsize=9,
               title='Planner', title_fontsize=10)

    plt.suptitle(f'Planner Comparison (step={step}) — {map_type} map', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'comparison_step={step}_{map_type}.png')
        plt.savefig(save_path, dpi=150)

    plt.show()


if __name__ == '__main__':
    # planner_names = ['A*', 'RRT*-Bi', 'RRT*-DBVSB-APF-Informed', 'VRRT*-Bi', 'VRRT*-Bi-Bias-APF-Informed']  # 规划器名称需手动提供
    # step = 15
    # save_dir = './test_results'
    
    # map_type = 'cluster'
    # avg_matrix = [
    #     [100.0,	0.469,	1720.0,	7.0,    0,      0.469,  1720.0,	7.0],
    #     [100.0,	0.051,	2004.2,	25.6,	368,	0.693,	1931.7,	22.1],
    #     [100.0,	0.040,	1993.1,	26.5,	305,	0.839,	1923.3,	22.5],
    #     [100.0,	0.017,	1800.2,	13.8,	123,	1.566,	1737.4,	7.3],
    #     [100.0,	0.015,	1751.8,	8.4,	107,	3.433,	1712.3,	2.7]]
    # std_matrix = [
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0.025, 97.300, 3.7, 156.0, 0.080, 54.800, 4.0],
    #     [0.013, 65.700, 3.4, 73.0, 0.041, 49.300, 3.9],
    #     [0.005, 31.2, 4, 16, 0.04, 6.2, 2.4],
    #     [0.002, 15.4, 2.7, 9, 0.187, 0.5, 0.7]]
    
    # map_type = 'clutter'
    # avg_matrix = [
    #     [100.0,	0.355,	1514.7,	16.8,    0,     0.355,  1514.7,	16.8],
    #     [99.5,	0.139,	1998.5,	24.9,	750,	0.644,	1892.1,	24.1],
    #     [100.0,	0.096,	1967.6,	25.6,	598,	0.711,	1817.1,	23.5],
    #     [100.0,	0.053,	1717.6,	16.7,	186,	2.096,	1625.9,	9.0],
    #     [100.0,	0.037,	1685.3,	14.2,	153,	4.241,	1608.0,	5.1]]
    # std_matrix = [
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0.060, 178.4, 3.6, 316, 0.099, 130.8, 4.4],
    #     [0.040, 165.2, 3.4, 234, 0.104, 74, 5],
    #     [0.028, 40.3, 4.4, 52, 0.07, 4.7, 3],
    #     [0.010, 29, 3.6, 24, 0.201, 1.8, 1.5]]
    
    # map_type = 'maze'
    # avg_matrix = [
    #     [100.0,	0.424,	1901.3,	2.2,    0,      0.424,  1901.3,	2.2],
    #     [87.5,	0.151,	2248.9,	24.3,	755,	0.563,	2184.9,	22.1],
    #     [100.0,	0.100,	2251.6,	25.5,	402,	1.007,	2140.6,	20.9],
    #     [100.0,	0.042,	2071.8,	16.5,	141,	2.643,	1976.6,	6.9],
    #     [100.0,	0.044,	2019.6,	12.1,	135,	3.994,	1962.8,	4.2]]
    # std_matrix = [
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0.076, 104.1, 4.4, 462, 0.116, 105, 5],
    #     [0.053, 97.4, 3.6, 115, 0.048, 62.4, 3.9],
    #     [0.013, 30.4, 3.9, 21, 0.041, 3.8, 2.4],
    #     [0.021, 24.9, 3.4, 32, 0.111, 3.8, 1.5]]

    # # 解包
    # success_rates = [row[0] for row in avg_matrix]
    # avg_data = {
    #     'max_time_first': [row[1] for row in avg_matrix],
    #     'len_first': [row[2] for row in avg_matrix],
    #     'smooth_first': [row[3] for row in avg_matrix],
    #     'sum_iter': [row[4] for row in avg_matrix],
    #     'time_final': [row[5] for row in avg_matrix],
    #     'len_final': [row[6] for row in avg_matrix],
    #     'smooth_final': [row[7] for row in avg_matrix]
    # }
    # err_data = {
    #     'max_time_first': [row[0] for row in std_matrix],
    #     'len_first': [row[1] for row in std_matrix],
    #     'smooth_first': [row[2] for row in std_matrix],
    #     'sum_iter': [row[3] for row in std_matrix],
    #     'time_final': [row[4] for row in std_matrix],
    #     'len_final': [row[5] for row in std_matrix],
    #     'smooth_final': [row[6] for row in std_matrix]
    # }

    # # 调用绘图函数
    # plot_planner_comparison(planner_names, success_rates, avg_data, err_data,
    #                         step, map_type, save_dir)
    
    algo_names = ['A*', 'RRT*-Bi', 'RRT*-DBVSB-APF-Informed', 'VRRT*-Bi', 'VRRT*-Bi-Bias-APF-Informed']  # 规划器名称需手动提供
    avg_cluster = [
        [100.0,	0.469,	1720.0,	7.0,    0,      0.469,  1720.0,	7.0],
        [100.0,	0.051,	2004.2,	25.6,	368,	0.693,	1931.7,	22.1],
        [100.0,	0.040,	1993.1,	26.5,	305,	0.839,	1923.3,	22.5],
        [100.0,	0.017,	1800.2,	13.8,	123,	1.566,	1737.4,	7.3],
        [100.0,	0.015,	1751.8,	8.4,	107,	3.433,	1712.3,	2.7]]
    
    avg_clutter = [
        [100.0,	0.355,	1514.7,	16.8,    0,     0.355,  1514.7,	16.8],
        [99.5,	0.139,	1998.5,	24.9,	750,	0.644,	1892.1,	24.1],
        [100.0,	0.096,	1967.6,	25.6,	598,	0.711,	1817.1,	23.5],
        [100.0,	0.053,	1717.6,	16.7,	186,	2.096,	1625.9,	9.0],
        [100.0,	0.037,	1685.3,	14.2,	153,	4.241,	1608.0,	5.1]]
    avg_maze = [
        [100.0,	0.424,	1901.3,	2.2,    0,      0.424,  1901.3,	2.2],
        [87.5,	0.151,	2248.9,	24.3,	755,	0.563,	2184.9,	22.1],
        [100.0,	0.100,	2251.6,	25.5,	402,	1.007,	2140.6,	20.9],
        [100.0,	0.042,	2071.8,	16.5,	141,	2.643,	1976.6,	6.9],
        [100.0,	0.044,	2019.6,	12.1,	135,	3.994,	1962.8,	4.2]]
    
    avg_matrices = [avg_cluster, avg_clutter, avg_maze]
    map_names = ['cluster', 'clutter', 'maze']
    
    # 调用绘图函数
    plot_algorithm_comparison_across_maps(avg_matrices, map_names, algo_names, save_dir='./test_results')
    

