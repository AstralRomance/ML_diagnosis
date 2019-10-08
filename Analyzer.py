from collections import Counter
from matplotlib import pyplot as plt

class Analyzer:
    def __init__(self, cluster_labels, data=None):
        self.points_per_cluster = Counter(cluster_labels)
        self.points_cluster_dict = data
        self.prob_per_cluster = []
        self.avg_cluster_probability = []

#   Probability calculating for clusters
    def probability_calc(self):
        total_points = sum(self.points_per_cluster.values())
        for i in self.points_per_cluster.keys():
            self.prob_per_cluster.append((self.points_per_cluster.get(i)/total_points)*100)
        self.__probability_print()

#   Counting math expectation for each component per clusters
#   Return list with expectation for each component per cluster
    def math_expectation(self):
        temp = 0
        math_expectation = []
        for clusters in self.points_cluster_dict.keys():
            temp_exp = []
            for x_component in self.points_cluster_dict[clusters][0]:
                temp += x_component
            temp_exp.append(temp/len(self.points_cluster_dict[clusters]))
            temp = 0
            for y_component in self.points_cluster_dict[clusters][1]:
                temp += y_component
            temp_exp.append(temp/len(self.points_cluster_dict[clusters]))
            math_expectation.append(math_expectation)
            temp_exp.clear()
        return math_expectation

#   Closed method to print probability per cluster
    def __probability_print(self):
        cnt = 0
        for i in self.prob_per_cluster:
            print(f'Probability for cluster {cnt} is {i}%')
            cnt += 1
        print('')
