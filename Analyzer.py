from collections import Counter
from matplotlib import pyplot as plt

class Analyzer:
    def __init__(self, data=None):
        #self.prob_per_cluster = []
        self.avg_cluster_probability = []
        self.prob_counter = [1, ]

#   Probability calculating for clusters
    def probability_calc(self, point_dict):
        total_points = sum(map(len, point_dict.values()))
        prob_per_cluster = [(len(point_dict.get(i))/total_points)*100 for i in point_dict.keys()]
        self.prob_counter.append(self.prob_counter[-1]+1)
        self.__probability_print(prob_per_cluster)
        return prob_per_cluster

    @property
    def get_prob_counter(self):
        return self.prob_counter

#   Closed method to print probability per cluster
    def __probability_print(self, probs):
        cnt = 0
        with open('probs.txt', 'a') as prob_f:
            for i in probs:
                print(f'Probability for cluster {cnt} is {i}%')
                prob_f.write(f'Probability for cluster {cnt} is {i}%')
                cnt += 1
            print('')
            prob_f.write('\n')

#   Counting math expectation for each component per clusters
#   Return list with expectation for each component per cluster
'''
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
'''



