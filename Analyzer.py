from collections import Counter
from matplotlib import pyplot as plt

class Analyzer:
    def __init__(self, cluster_labels, data=None):
        self.points_per_cluster = Counter(cluster_labels)
        self.prob_per_cluster = []

#   Probability calculating for clusters
    def probability_calc(self):
        total_points = sum(self.points_per_cluster.values())
        for i in self.points_per_cluster.keys():
            self.prob_per_cluster.append((self.points_per_cluster.get(i)/total_points)*100)
        self.__probability_print()

#   Closed method to print probability per cluster
    def __probability_print(self):
        cnt = 0
        for i in self.prob_per_cluster:
            print(f'Probability for cluster {cnt} is {i}%')
            cnt += 1
        print('')
