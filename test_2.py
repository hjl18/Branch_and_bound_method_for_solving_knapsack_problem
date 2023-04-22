from Knapsack import branch_and_bound
import Knapsack
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

def cal_error(queue_bound,queue_theory):
    n = len(queue_bound) # 获取两个数组的长度
    # 如果两个数组长度不一致,输出错误信息,返回
    if n != len(queue_theory):
        print('The length of two arrays is inconsistent!')
        return
    error = np.zeros(n)# 创建长度为n的误差数组,初始化为0
    # 遍历两个数组,计算每个元素的绝对误差,存储在误差数组中
    for i in range(n):
        error[i] = math.fabs(queue_bound[i] - queue_theory[i])
    return error


def test_2_const_n(n, capacity, num_samples):
    # 创建长度为n的上界队列、理论下界队列和计数数组,初始化为0
    queue_bound = np.zeros(n)
    queue_theory = np.zeros(n)
    count = np.zeros(n)

    # 进行num_samples次测试
    for _ in range(num_samples):

        # 初始化
        Knapsack.q = []
        Knapsack.best_value = 0
        Knapsack.branches = 0

        # 生成随机物品价值和重量
        values = [random.randint(1, 100) for _ in range(n)]
        weights = [random.randint(1, 100) for _ in range(n)]

        # 按照价值密度排序物品
        value_density = [v / w for v, w in zip(values, weights)]
        index = sorted(range(len(value_density)), key=lambda k: value_density[k], reverse=True)
        values = [values[i] for i in index]
        weights = [weights[i] for i in index]

        # 调用分支限界算法
        branch_and_bound(0, 0, 0, n, weights, capacity, values)

        # 按照层数升序排序队列
        index = sorted(range(len(Knapsack.q)), key=lambda k: Knapsack.q[k].level, reverse=False)

        # 累加每个层数的上界、理论下界和计数
        for i in index:
            queue_bound[Knapsack.q[i].level] += Knapsack.q[i].bound
            queue_theory[Knapsack.q[i].level] += Knapsack.q[i].theory_value
            count[Knapsack.q[i].level] += 1

     # 求平均上界和理论下界
    for i in range(n):
        queue_bound[i] /= count[i]
        queue_theory[i] /= count[i]

    # 计算误差
    error = cal_error(queue_bound, queue_theory)

    # 返回平均上界、理论下界和误差
    return queue_bound, queue_theory, error


def test_2_change_n(n1, n2, capacity):
    # 创建(n1+1)行n2列的上界队列和理论下界队列,初始化为0
    queue_bound = np.zeros((n1 + 1, n2))
    queue_theory = np.zeros((n1 + 1, n2))

    # 从n1到n2个物品
    for n in range(n1, n2 + 1):

        # 初始化
        Knapsack.q = []
        Knapsack.best_value = 0
        Knapsack.branches = 0

        # 生成随机物品价值和重量
        values = [random.randint(1, 100) for _ in range(n)]
        weights = [random.randint(1, 100) for _ in range(n)]

        # 按照价值密度排序物品
        value_density = [v / w for v, w in zip(values, weights)]
        index = sorted(range(len(value_density)), key=lambda k: value_density[k], reverse=True)
        values = [values[i] for i in index]
        weights = [weights[i] for i in index]

        # 调用分支限界算法
        branch_and_bound(0, 0, 0, n, weights, capacity, values)

        # 按照层数升序排序队列
        index = sorted(range(len(Knapsack.q)), key=lambda k: Knapsack.q[k].level, reverse=False)

        # 存储每个层数的上界和理论下界
        for i in index:
            queue_bound[n - 10][Knapsack.q[i].level] = Knapsack.q[i].bound
            queue_theory[n - 10][Knapsack.q[i].level] = Knapsack.q[i].theory_value

            # 计算上界和理论下界的绝对误差
    error = np.fabs(queue_bound - queue_theory)

    # 返回误差,可以画误差随物品数变化的曲线图
    return error

#可视化函数
def showGraph_const_n(n,capacity,num_samples):
    # 调用test_2_const_n进行算法误差测试,得到平均上界、理论下界和误差
    queue_bound ,queue_theory,error = test_2_const_n(n,capacity,num_samples)
    x = [i for i in range(len(queue_bound))]
    plt.title('Cost Function and Real Value(n = 10)')
    plt.plot(x, queue_bound, 'ro-', label='Estimated value')
    plt.plot(x, queue_theory, 'bo-', label='Real Value')
    plt.xlabel('Level')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.title('Absolute error(n = 10)')
    plt.plot(x,error,'ro-')
    plt.xlabel('Level')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

def showGraph_change_n(n1,n2,capacity):
    # y轴刻度标签格式化函数
    def y_tick_formatter(y,pos):
        return '{:.0f}'.format(y + 10)

    # 调用test_2_change_n进行算法误差测试,得到误差
    error = test_2_change_n(n1,n2,capacity)
    x = np.array([(i - 10) for i in range(n1,n2 + 1)])
    y = np.array([i for i in range(n1)])
    # 构建 deux 网格坐标系
    X, Y = np.meshgrid(y, x)#X为level,Y为n
    z = np.array(error)
    Z = z[:,:10]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X,Y,Z, cmap='viridis')
    ax.set_xlabel('Level')
    ax.set_ylabel('n')
    ax.set_zlabel('Absolute error')
    # y轴刻度标签格式化
    ax.yaxis.set_major_formatter(y_tick_formatter)
    plt.show()

