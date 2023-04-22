from scipy.optimize import linprog
import numpy as np
n = 6
C = 11

# 物品重量和价值
weights = [2, 3, 4, 5, 6, 7]
values = [3, 4, 5, 6, 7, 8]
value = [-i for i in values]
best_sol = 0
bounds = [(0, None)] * n
# 引入松弛变量
x = [0] * n

# 目标函数
obj_func = sum(x[i] * values[i] for i in range(n))

# 约束条件
constraints = [sum(x[i] * weights[i] for i in range(n)) <= C]
weight = list()
weight.append(weights)
b_ub = [C]
# 记录上界最小的节点
best_node = None
while True:
    # 求解松弛问题,得到一个比较好的下界
    res = linprog(c= value, A_ub=weight, b_ub=b_ub, bounds=bounds)
    x = res.x
    #x = x.astype(int)
    obj_func = sum(x[i] * values[i] for i in range(n))
    # 上界 = 目标函数值 + 剩余空间内最大价值
    upper_bound = obj_func + (C - sum(x[i] * weights[i] for i in range(n))) * max(values)
    print(upper_bound,best_sol)
    # 如果上界<=最优解,则找到最优解,结束搜索
    if upper_bound >= best_sol:
        best_sol = obj_func
        break

        # 选择代价(上界)最小的节点进行展开
    if best_node is None or upper_bound < best_node[0]:
        best_node = (upper_bound, x[:])

        # 展开最有希望的节点
    if x[0] < weights[0]:
        x[0] += 1  # 选择第一个物品
    else:
        idx = 1
        while x[idx] >= weights[idx] and idx < n:  # 找第一个可以增量的变量
            x[idx] = 0
            idx += 1
        x[idx] += 1  # 增量并继续搜索

print(int(best_sol))