from Knapsack import branch_and_bound,new_branch_and_bound
import Knapsack
import random
import matplotlib.pyplot as plt
import numpy as np
import math

def cal_error(queue_bound,queue_theory):
    n = len(queue_bound)
    if n != len(queue_theory):
        print('The length of two arrays is inconsistent!')
        return
    error = np.zeros(n)
    for i in range(n):
        error[i] = math.fabs(queue_bound[i] - queue_theory[i])
    return error


def test_3(n, capacity, num_samples):
    # 创建长度为n的两种算法的上界队列、理论下界队列和计数数组,初始化为0
    queue_bound = np.zeros(n)
    queue_theory = np.zeros(n)
    count = np.zeros(n)

    queue_bound_new = np.zeros(n)
    queue_theory_new = np.zeros(n)
    count_new = np.zeros(n)

    # 进行num_samples次测试
    for _ in range(num_samples):

        # 生成随机物品价值和重量
        values = [random.randint(1, 100) for _ in range(n)]
        weights = [random.randint(1, 100) for _ in range(n)]

        # 按照价值密度排序物品
        value_density = [v / w for v, w in zip(values, weights)]
        index = sorted(range(len(value_density)), key=lambda k: value_density[k], reverse=True)
        values = [values[i] for i in index]
        weights = [weights[i] for i in index]

        # 调用第一种分支限界算法
        Knapsack.q = []
        Knapsack.best_value = 0
        Knapsack.branches = 0
        branch_and_bound(0, 0, 0, n, weights, capacity, values)
        index = sorted(range(len(Knapsack.q)), key=lambda k: Knapsack.q[k].level, reverse=False)

        # 累加每个层数的上界、理论下界和计数
        for i in index:
            queue_bound[Knapsack.q[i].level] += Knapsack.q[i].bound
            queue_theory[Knapsack.q[i].level] += Knapsack.q[i].theory_value
            count[Knapsack.q[i].level] += 1

            # 调用第二种分支限界算法
        Knapsack.q = []
        Knapsack.best_value = 0
        Knapsack.branches = 0
        new_branch_and_bound(0, 0, 0, n, weights, capacity, values)
        index = sorted(range(len(Knapsack.q)), key=lambda k: Knapsack.q[k].level, reverse=False)

        # 累加每个层数的上界、理论下界和计数
        for i in index:
            queue_bound_new[Knapsack.q[i].level] += Knapsack.q[i].bound
            queue_theory_new[Knapsack.q[i].level] += Knapsack.q[i].theory_value
            count_new[Knapsack.q[i].level] += 1

            # 求两种算法的平均上界和理论下界
    for i in range(n):
        queue_bound[i] /= count[i]
        queue_theory[i] /= count[i]

        queue_bound_new[i] /= count_new[i]
        queue_theory_new[i] /= count_new[i]

    # 分别计算两种算法的误差
    error = cal_error(queue_bound, queue_theory)
    error_new = cal_error(queue_bound_new, queue_theory_new)

    # 返回两种算法的误差
    return error, error_new

#可视化函数
def showGraph(n,capacity,num_samples):
    error,error_new = test_3(n,capacity,num_samples)
    x = [i for i in range(len(error))]
    plt.title('Comparison between the new cost function and the initial cost function')
    plt.plot(x,error,'ro-',label = 'The initial cost function')
    plt.plot(x,error_new,'bo-',label = 'The new cost function')
    plt.xlabel('Level')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()