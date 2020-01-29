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

    def normal_check(self, dataset):
        return stats.shapiro(dataset), stats.normaltest(dataset)

    def metric_collection(self, estimator_name, metric_set):
        if estimator_name == 'KMeans':
            self.metric_dataset = pd.DataFrame(metric_set, columns=['number of clusters', 'train dataset length', 'test dataset length',
                                                                     'silhouette metric', 'calinski metric', 'david bolduin metric'])
            self.metric_dataset.to_csv(f'metrics/kmeans_clustering.csv')

    def best_clustering_find(self):
        best_metrics = list(
                                zip(
                                    ('silhouette metric', 'calinski metric', 'david bolduin metric'),

                                    (max(self.metric_dataset['silhouette metric']), max(self.metric_dataset['calinski metric']),
                                        min(self.metric_dataset['david bolduin metric']))
                                    )
                            )
        for ind, metric in best_metrics:
            print(self.metric_dataset[self.metric_dataset[ind] == metric])

    def get_metric_dataset(self):
        return self.metric_dataset

    #   Closed method to print probability per cluster
    def __probability_print(self, label,  prob):
        print(f'Probability in {label} cluster = {prob}')

