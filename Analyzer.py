from collections import Counter
from matplotlib import pyplot as plt

'''
Class for distribution analysis
'''


class Analyzer:
    def __init__(self):
        self.avg_cluster_probability = []
        self.probability_list = []

#   Probability calculating for clusters
    def probability_calc(self, point_dict):
        total_points = sum(map(len, point_dict.values()))
        prob_per_cluster = [(len(point_dict.get(i))/total_points)*100 for i in point_dict.keys()]
        self.__probability_print(prob_per_cluster)
        self.probability_list.append(prob_per_cluster)

#   Counting math expectation for each component per clusters
#   Return list with expectation for each component per cluster

    def math_expectation(self, cluster_dict):
        temp = [0, 0]
        sum_dict = {}
        res = []
        points_per_cluster = list(map(len, cluster_dict.values()))

        for clusters in cluster_dict.keys():
            for points in cluster_dict.get(clusters):
                temp[0] += points[0]
                temp[1] += points[1]
            sum_dict[clusters] = temp
        for clusters in sum_dict.keys():
            res.append([sum_dict.get(clusters)[0]/points_per_cluster[clusters],
                        sum_dict.get(clusters)[1]/points_per_cluster[clusters]])
        return res

    def pred_start_difference(self, real_data, predicted):
        t = []
        with open('Linear regression difference.txt', 'w') as diff_file:
            for i, j in enumerate(real_data):
                diff_file.write(str(real_data[i]-predicted[i]) + '\n')
                print(str(real_data[i]-predicted[i]))
                t.append(real_data[i]-predicted[i])
        return t

    @property
    def get_total_probability(self):
        return self.probability_list

#   Closed method to print probability per cluster

    def __probability_print(self, probs):
        cnt = 0
        with open('probs.txt', 'a') as prob_f:
            for i in probs:
                print(f'Probability for cluster {cnt} is {i}%')
                prob_f.write(f'Probability for cluster {cnt} is {i}%  \n')
                cnt += 1
            print('')
            prob_f.write('\n\n')
