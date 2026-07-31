# 面向展示的测评

本文件夹用于给出面向展示的测评数据，可在杂乱、簇状、迷宫环境中对A*、Bi-RRT*、RRT*-DBVSB-APF-Informed、Bi-VRRT*、VRRT*-Bi-Bias-APF-Informed五种三维路径规划器进行性能测试。

## 文件结构

```
result_for_show/
├── test_results/                                # 统计图结果记录
├── env_generator_for_data.py                    # 环境生成器
├── planner_A_star.py                            # A*算法规划器
├── planner_RRT_star_Bi.py
├── planner_RRT_star_Bi_DBVS_APF_Informed.py
├── planner_VRRT_star_Bi.py
├── planner_VRRT_star_Bi_Bias_APF_Informed.py
├── res_show.py                                  # 绘制环境或路径
├── test_case_clutter.pkl                        # 杂乱环境测试用例
├── test_case_cluster.pkl                        # 簇状环境测试用例
├── test_case_maze.pkl                           # 迷宫环境测试用例
└── test1_baseline.py                            # RRT算法测试脚本
```

## 使用方法

确保所有 `.py` 文件与 `.pkl` 文件位于同一目录。在 `test1_baseline.py` 的主函数开头部分设置以下参数：
   - `map_type`：选择 `'homogeneous'`、`'cluster'` 或 `'maze'`
   - `planner_type_list`：指定要测试的规划器类型（例如 `['RRT_star_Bi','VRRT_star_Bi']`）
   - `num_of_tests`：每种规划器的重复测试次数
   - `step`：扩展步长
   - `max_iter`：RRT 算法最大迭代次数
   - `search_radius`：RRT 算法搜索半径

在终端输入 `python3 test1_baseline.py` 运行测试脚本，程序将：
   - 显示地图与航路点
   - 按`planner_type_list`的顺序，依次对规划器重复进行`num_of_tests`次测试
   - 收集成功率、首次规划时间、首次路径长度、首次路径平滑度、总迭代次数、最终规划时间、最终路径长度、最终路径平滑度共8个指标，并计算后7个指标的标准差
   - 在 `test_results/` 目录下保存png格式的对比柱状图
   - 在终端打印各个规划器的8个指标平均值与7个指标标准差

特别注意：由于A*算法返回结果仅包括首次规划时间、首次路径长度、首次路径平滑度，且该算法为确定性算法，即在给定参数与环境下路径完全固定（规划时间受设备影响可能出现轻微偏差），在数据收集与绘图部分与RRT测评脚本不兼容，因此`test1_baseline.py`不能对该算法进行测试。需要在终端输入`python3 planner_A_star.py`得到算法的首次规划时间、首次路径长度、首次路径平滑度参数。

## 输出指标

- **Success Rate (%)** – 成功找到完整路径的比例
- **Time First (s)** – 首次发现可行路径的耗时（各航段耗时的最大值）
- **Length First** – 首次路径的总长度
- **Smoothness First (deg)** – 首次路径的平滑度
- **Iterations Total** – 总迭代次数（各航段迭代次数之和）
- **Time Final (s)** – 完成最终优化（或达到最大迭代）的总耗时
- **Length Final** – 最终路径长度
- **Smoothness Final (deg)** – 最终路径的平滑度

柱状图中，`Length Final` 和 `Smoothness Final` 上方会标注相比 `First` 的缩减百分比。

## 结果示例
输入`python3 test1_baseline.py`将输出下述类似结果：
```
规划器类型: ['RRT_star_Bi', 'VRRT_star_Bi']
地图类型: maze
测试步长: 15
每个步长重复规划 2 次
1
2
1
2
RRT_star_Bi               | 成功: 1/2 | 成功率: 50.0%
VRRT_star_Bi              | 成功: 2/2 | 成功率: 100.0%

================================================================================
  平均指标 (Average Metrics)
================================================================================
Planner                     Success Rate(%)     Time First(s)         Len First Smooth First(deg)        Iter Total     Time Final(s)         Len Final Smooth Final(deg)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
RRT_star_Bi                            95.0             0.161            2276.2              24.4               764             0.608            2189.0              21.5
VRRT_star_Bi                          100.0             0.047            2073.4              17.2               144             2.838            1975.9               6.8

================================================================================
  标准差 (Standard Deviation)
================================================================================
Planner                       Time First(s)         Len First Smooth First(deg)        Iter Total     Time Final(s)         Len Final Smooth Final(deg)
-------------------------------------------------------------------------------------------------------------------------------------------------------
RRT_star_Bi                           0.080             139.0               4.0               501             0.116             120.9               4.9
VRRT_star_Bi                          0.035              32.7               4.8                49             0.064               3.2               1.5
```

输入`python3 planner_A_star.py`将输出下述类似结果：
```
========== 规划结果 ==========
路径规划耗时: 0.417 秒
路径长度: 1901.28 单位
路径步数: 99
路径平滑度: 2.16 度
扩展节点数: 41166
起点栅格索引: (0, 16, 8)
终点栅格索引: (99, 83, 8)
==============================
```

**日期**：2026.07.29
