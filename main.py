import MonteCarlo
import Knapsack
import test_1,test_2,test_3

if __name__ == '__main__':
    n = 15
    capacity = 100
    num_samples = 100
    Knapsack.q = []
    Knapsack.branches = 0
    Knapsack.best_value = 0
    test_1.showGraph(n,capacity,num_samples)
    MonteCarlo.showGraph(capacity,num_samples)
    test_2.showGraph_const_n(n,capacity,num_samples)
    n1 = 10
    n2 = 20
    test_2.showGraph_change_n(n1,n2,capacity)
    test_3.showGraph(n,capacity,num_samples)
