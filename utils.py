import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime

def plot_running_averages(data_list, labels, step, figsize=(18, 5),
                               ylim_expand=0.3, save_dir='plots'):
    """
    横向排列绘制多条累积平均值曲线
    ----------
    data_list :  待绘制的指标数据列表
    labels :     每个子图的标题与y轴标签
    step :       取样步长
    figsize :    图形尺寸
    ylim_expand: 纵轴上下扩展比例
    save_dir :   保存图片的子文件夹名称
    """
    # 获取运行脚本的文件名
    script_path = sys.argv[0]
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.splitext(os.path.basename(script_path))[0]

    # 创建保存目录
    save_path_dir = os.path.join(script_dir, save_dir)
    os.makedirs(save_path_dir, exist_ok=True)

    # 文件名
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels_short = '_'.join([lbl.replace(' ', '_') for lbl in labels])
    file_name = f"{script_name}_{labels_short}_{time_str}.png"
    save_path = os.path.join(save_path_dir, file_name)

    # 绘图
    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, data, label in zip(axes, data_list, labels):
        total = len(data)
        if total < step:
            ax.set_title(f'{label}\n(数据不足{step}次)')
            ax.set_xlabel('Number of Plans')
            ax.set_ylabel(label)
            continue

        n_points = (total // step) * step
        truncated = data[:n_points]

        arr = np.array(truncated)
        cum_avg = np.cumsum(arr) / np.arange(1, n_points + 1)

        x_ticks = np.arange(step, n_points + 1, step)
        y_avg = cum_avg[step - 1::step]

        ax.plot(x_ticks, y_avg, marker='o', linestyle='-')
        ax.set_xlabel('Number of Plans')
        ax.set_ylabel(label)
        ax.set_title(f'Cumulative Average of {label}')
        ax.grid(True)

        # 放大纵轴范围
        ymin, ymax = np.min(y_avg), np.max(y_avg)
        y_range = ymax - ymin if ymax > ymin else abs(ymin) * 0.1 or 1.0
        ax.set_ylim(ymin - y_range * ylim_expand,
                    ymax + y_range * ylim_expand)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)