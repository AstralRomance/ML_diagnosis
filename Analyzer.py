from scipy import stats
import pandas as pd


'''
Class for distribution analysis
'''

class Analyzer:
    def __init__(self, input_data=None):
        self.avg_cluster_probability = []
        self.metric_dataset = None

#   Probability calculating per clusters
    def probability_per_cluster(self, cluster_len, total_len, cluster_label):
        prob_per_cluster = (cluster_len/total_len)*100
        self.__probability_print(cluster_label, prob_per_cluster)
        return prob_per_cluster

#   Returns local extreme points
    def extreme_found(self, cluster_dict):
        extreme_max = {}
        extreme_min = {}
        for cluster in cluster_dict:
            extreme_max[cluster] = max(cluster_dict.get(cluster))
            extreme_min[cluster] = min(cluster_dict.get(cluster))
        return (extreme_max, extreme_min)

    def normal_check(self, dataset):
        return stats.shapiro(dataset), stats.normaltest(dataset)

    def metric_collection(self, estimator_name, metric_set):
        if estimator_name == 'KMeans':
            self.metric_dataset = pd.DataFrame(metric_set, columns=['train dataset length', 'test dataset length',
                                                                     'silhouette metric', 'calinski metric', 'david bolduin metric'])
            self.metric_dataset.to_csv('metrics/kmeans_clustering.csv')

    def best_clustering_find(self):
        best_metrics = list(
                                zip(
                                    ('silhouette metric', 'calinski metric', 'david bolduin metric'),

                                    (max(self.metric_dataset['silhouette metric']), max(self.metric_dataset['calinski metric']),
                                        min(self.metric_dataset['david bolduin metric']))
                                    )
                            )
        for k, v in best_metrics:
            print(self.metric_dataset[self.metric_dataset[k] == v])




#   Closed method to print probability per cluster
    def __probability_print(self, label,  prob):
        print(f'Probability in {label} cluster = {prob}')
