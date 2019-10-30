from scipy import stats


'''
Class for distribution analysis
'''


class Analyzer:
    def __init__(self, input_data):
        self.avg_cluster_probability = []
        self.probability_list = []
        self.input_data = input_data['К0011'].values
        self.input_data.sort()
        print(self.input_data)

#   Probability calculating for clusters
    def probability_per_cluster(self, point_dict):
        total_points = sum(map(len, point_dict.values()))
        prob_per_cluster = [(len(point_dict.get(i))/total_points)*100 for i in point_dict.keys()]
        self.__probability_print(prob_per_cluster)
        self.probability_list.append(prob_per_cluster)

#   Counting math expectation for each component per clusters
#   Return list with expectation for each component per cluster

    def math_expectation(self, cluster_dict):
        res = []
        for clusters in cluster_dict.keys():
            #print(group_sum)
            res.append(sum(cluster_dict[clusters])/len(cluster_dict.get(clusters)))
        return res

    def dispersion(self, cluster_dict):
        res = []
        temp = 0
        for cluster in cluster_dict:
            cluster_len = len(cluster_dict[cluster])
            mid = sum(cluster_dict[cluster])/cluster_len
            for point in cluster_dict[cluster]:
                temp += (point - mid)**2
            res.append(temp/cluster_len)
            temp = 0
        return res

    def extreme_found(self, cluster_dict):
        extreme_max = {}
        extreme_min = {}
        for cluster in cluster_dict:
            extreme_max[cluster] = max(cluster_dict.get(cluster))
            extreme_min[cluster] = min(cluster_dict.get(cluster))
        return (extreme_max, extreme_min)

# РАБОТАЕТ СТРАННО, ОБЩАЯ ВЕРОЯТНОСТЬ НИЖЕ 1, ИНОГДА СИЛЬНО!!!
    def make_intervals(self, step):
        start_interval = min(self.input_data)
        finish_interval = start_interval + step
        #print(start_interval)
        current_step = step
        res_dict = {}
        temp = []
        while current_step <= max(self.input_data):
            for j in self.input_data:
                if (j >= start_interval) and (j < finish_interval):
                    temp.append(j)
                else:
                    continue
            res_dict[len(temp)/len(self.input_data)] = temp
            start_interval = finish_interval
            finish_interval += step
            current_step += step
            temp = []
        del res_dict[0.0]
        return res_dict

    def normal_check(self, dataset):
        return stats.shapiro(dataset), stats.normaltest(dataset)

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
