# Branch_and_bound_method_for_solving_knapsack_problem
该实验的主要文件有：

- Knapsack.py
- MonteCarlo.py
- test_1.py
- test_2.py
- test_3.py
- main.py

各文件主要实现的功能：

- Knapsack.py
  - 回溯法
  - 动态规划法
  - 分支限界法
  - 改进后的分支限界法
- MonteCarlo.py
  - 蒙特卡洛法对分支数量进行估计
- test_1.py
  - 以物品种类数n为输入规模，固定n，随机产生大量测试样本
  - 用回溯法和分支限界法运行100组测试
- test_2.py
  - 记录结点的代价函数与真实值，分析在同一输入规模下不同层代价函数的近似效果
  - 分析在不同输入规模下同一层代价函数的近似效果
- test_3.py
  - 改变代价函数前后两种分支限界法所得结果与真实值之间的误差

该实验在WSL2环境中运行。

使用前运行命令：`sudo apt-get install python3-tk `确保在命令行运行程序时可以将图像显示出来。

运行命令：`python --version`查看是否在环境变量中配置好python环境，该实验使用了python3，如果安装的是python2请在官网上安装好python3。

运行命令：`python main.py` 即可将实验结果可视化。
