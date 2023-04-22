from Knapsack import branch_and_bound
import Knapsack
import random
import matplotlib.pyplot as plt

def Monte_Carlo(n,capacity,num_samples):
    total_branches = []# 记录每次求解的分支数
    for i in range(num_samples):# 进行num_samples次测试
        # 初始化
        Knapsack.q = []
        values = [random.randint(1, 100) for _ in range(n)]
        weights = [random.randint(1, 100) for _ in range(n)]
        value_density = [v / w for v, w in zip(values, weights)]
        # 按照价值密度排序物品
        index = sorted(range(len(value_density)), key=lambda k: value_density[k], reverse=True)
        values = [values[i] for i in index]
        weights = [weights[i] for i in index]
        # 初始化最优解和分支数
        Knapsack.branches = 0
        Knapsack.best_value = 0
        # 调用分支限界算法求解
        branch_and_bound(0, 0, 0, n, weights, capacity, values)
        # 记录这次求解的分支数
        total_branches.append(Knapsack.branches)
    # 计算平均分支数
    avg_branches = sum(total_branches) / num_samples
    return avg_branches

def showGraph(capacity,num_samples):
    branches = [] # 记录10到20个物品情况下的平均分支数
    for n in range(10,21):
        b = Monte_Carlo(n,capacity,num_samples) # 调用蒙特卡罗算法,进行num_samples次求解,得到平均分支数
        branches.append(b)
    x = [i for i in range(10,21)]
    plt.title('The number of branches')
    plt.plot(x, branches, 'ro-')
    plt.xlabel('n')
    plt.ylabel('Number')
    plt.grid(True)
    plt.show()

