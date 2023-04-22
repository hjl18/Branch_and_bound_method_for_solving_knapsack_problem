from Knapsack import backtrack,branch_and_bound,dp
import Knapsack
import random
import time
import matplotlib.pyplot as plt
import numpy as np

def test_1(n,capacity,num_samples):
    T1 = []# 记录回溯算法的运行时间
    T2 = [] # 记录分支限界算法的运行时间
    Knapsack.q = []
    for i in range(num_samples):# 进行num_samples次测试
        # 生成随机物品价值和重量
        values = [random.randint(1, 100) for _ in range(n)]
        weights = [random.randint(1, 100) for _ in range(n)]
        # 初始化最优解和分支数
        Knapsack.best_value = 0
        Knapsack.branches = 0
        start_time = time.perf_counter() # 记录开始时间
        backtrack(0,0,0,n,weights,capacity,values)# 调用回溯算法求解
        end_time = time.perf_counter()# 记录结束时间
        T1.append( (end_time - start_time) * 1000) # 计算运行时间并记录
        # 初始化最优解和分支数
        Knapsack.best_value = 0
        Knapsack.branches = 0
        start_time = time.perf_counter()# 记录开始时间
        branch_and_bound(0, 0, 0, n, weights, capacity, values) # 调用分支限界算法求解
        end_time = time.perf_counter() # 记录结束时间
        T2 .append((end_time - start_time) * 1000)# 计算运行时间并记录
    # 计算回溯算法的平均运行时间
    sum1 = sum(T1)
    print('The average time of backtrack is ',sum1 / num_samples,' ms')
    # 计算分支限界算法的平均运行时间
    sum2 = sum(T2)
    print('The average time of branch and bound is ',sum2 / num_samples,' ms')
    return T1,T2

def showGraph(n,capacity,num_samples):#将运行时间可视化
    T1,T2 = test_1(n,capacity,num_samples)
    x = [i for i in range(num_samples)]
    plt.title('The elapsed time')
    plt.plot(x,T1,'ro-',label='Backtrack')
    plt.plot(x,T2,'bo-',label='Branch and Bound')
    plt.xlabel('Group')
    plt.ylabel('Time(ms)')
    plt.grid(True)
    plt.legend()
    plt.show()

