import pstats
import pstats
# 加载性能数据
p = pstats.Stats('output.prof')

# 按累计时间（包含子函数调用）降序排列，打印前 30 个最耗时的函数
print("\n===== 按累计时间 (Cumulative Time) 排序 =====\n")
p.sort_stats('cumtime').print_stats(10)

# 按函数自身执行时间（不包含子调用，如纯CPU计算）降序排列
print("\n===== 按自身时间 (Internal Time) 排序 =====\n")
p.sort_stats('tottime').print_stats(10)