import numpy as np
import random
best_value = 0# 当前最优解
branches = 0 # 分支数
q = []
def backtrack(i,v,w,n,weights,capacity,values):
    global best_value,branches# 使用全局变量best_value和branches
    branches += 1# 分支数加1
    if i >= n:# 如果已经处理到最后一个物品
        if v > best_value and w <= capacity: # 如果当前价值更优并且重量不超过背包容量
            best_value = v # 更新最优解
        return # 回溯
    # 试探第i个物品的不同取值
    num = (capacity - w) // weights[i]# 能取的最大数量
    while num >= 0: # 从最大数量开始试探
        w1 = w + num * weights[i]# 当前重量
        if w1 > capacity:# 如果超过容量,不能取这么多
            continue
        v1 = v + num * values[i] # 当前价值
        backtrack(i + 1,v1,w1,n,weights,capacity,values)# 递归求解
        num -= 1 # 尝试取更少的第i个物品
'''
其中，i表示当前考虑到第i个物品，v表示当前价值，w表示当前重量，best_value表示最优解。
'''

class Node:
    def __init__(self,level,bound,theory_value):
        self.level = level#记录节点的深度
        self.bound = bound#记录节点的估计代价
        self.theory_value = theory_value#记录节点的真实值

# 分支限界算法求解整数背包问题
def branch_and_bound(i,w,v,n,weights,capacity,values):
    global best_value,branches # 使用全局变量best_value和branches
    # base case: 如果已经处理到最后一个物品
    if i == n :
        if v > best_value and w <= capacity:# 如果当前价值更优并且重量不超过背包容量
            best_value = v # 更新最优解
        return
    # 计算当前分支的上界和理论下界
    bound = v + (capacity - w) * (values[i] / weights[i]) # 上界
    theory_value,_ = dp(n - i,weights[i:],capacity - w,values[i:])# 理论下界
    theory_value += v #真实值计算
    # 剪枝: 如果上界小于最优解,剪掉这条分支
    if bound < best_value:
        return
    u = Node(i, bound, theory_value)#仅记录未被剪枝的结点的值
    q.append(u)
    # 遍历第i个物品的所有可能取值
    num = (capacity - w) // weights[i]
    while num >= 0:
        branches += 1 # 分支数加1
        w1 = w + num * weights[i] # 当前重量
        if w1 > capacity:# 如果超过容量,不能取这么多
            continue
        v1 = v + num * values[i]# 当前价值
        # 递归寻找下一物品的最优解
        branch_and_bound(i + 1,w1,v1,n,weights,capacity,values)
        num -= 1 # 尝试更少的物品


def new_branch_and_bound(i,w,v,n,weights,capacity,values):
    global best_value,branches # 使用全局变量best_value和branches
    # base case: 如果已经处理到最后一个物品
    if i == n :
        if v > best_value and w <= capacity:# 如果当前价值更优并且重量不超过背包容量
            best_value = v  # 更新最优解
        return
    # 计算当前分支的上界
    bound = v
    w_tem = w # 临时重量
    if w_tem != capacity:# 如果不超重
        for j in range(i,n):# 从当前物品到最后一个物品
            while w_tem + weights[j] <= capacity:# 添加重量不超过容量的物品
                bound += values[j]
                w_tem += weights[j]
        bound += (capacity - w_tem) * values[i] / weights[i]# 剩余容量添加当前物品
    # 计算理论下界
    theory_value,_ = dp(n - i,weights[i:],capacity - w,values[i:])
    theory_value += v #真实值计算
    # 剪枝: 如果上界小于最优解,剪掉这条分支
    if bound < best_value:
        return
    u = Node(i, bound, theory_value)#仅记录未被剪枝的结点的值
    q.append(u)
    # 遍历第i个物品的所有可能取值
    num = (capacity - w) // weights[i]
    while num >= 0:
        branches += 1  # 分支数加1
        w1 = w + num * weights[i]  # 当前重量
        if w1 > capacity:  # 如果超过容量,不能取这么多
            continue
        v1 = v + num * values[i]  # 当前价值
        # 递归寻找下一物品的最优解
        new_branch_and_bound(i + 1, w1, v1, n, weights, capacity, values)
        num -= 1  # 尝试更少的物品

def dp(n,weights,capacity,values):
    solution = np.zeros((n + 1,capacity + 1))# 创建n+1行和capacity+1列的数组,初始化为0
    # 外层循环遍历物品,内层循环遍历容量
    for i in range(1,n+1):
        for j in range(capacity + 1):
            # 如果当前容量j无法装入第i个物品,那么当前值等于上一行同一列的值
            if j < weights[i - 1]:
                solution[i][j] = solution[i-1][j]
            # 否则,当前值是上一行同一列的值和上一行容量为j-当前物品重量那一列的值+当前物品价值的最大值
            else:
                solution[i][j] = max(solution[i-1][j],solution[i][j - weights[i - 1]] + values[i - 1])
    return solution[-1][-1],solution # 返回n行capacity列的值,即最大价值和建立好的解表

if __name__ == '__main__':
    n = 10
    capacity = 100
    values = [random.randint(1, 100) for _ in range(n)]
    weights = [random.randint(1, 100) for _ in range(n)]
    best_value = 0
    backtrack(0,0,0,n,weights,capacity,values)
    print(best_value)
    value_density = [v/w for v, w in zip(values, weights)]
    index = sorted(range(len(value_density)), key=lambda k: value_density[k], reverse=True)
    values = [values[i] for i in index]
    weights = [weights[i] for i in index]
    best_value = 0
    new_branch_and_bound(0,0,0,n,weights,capacity,values)
    print(best_value)
    b,_ = dp(n,weights,capacity,values)
    print(b)
    best_value = 0
    branch_and_bound(0, 0, 0, n, weights, capacity, values)

