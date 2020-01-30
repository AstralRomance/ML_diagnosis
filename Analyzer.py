from scipy import stats
import pandas as pd

'''
Class for distribution analysis
'''

class Analyzer:
    def __init__(self, input_data=None):
        self.metric_dataset = None
        self.clusters_list = []

    def normal_check(self):
        for cluster in self.clusters_list:
            for predictor_list in cluster:
                yield stats.shapiro(cluster[predictor_list]), stats.normaltest(cluster[predictor_list])

    def metric_collection(self, estimator_name, metric_set):
        if estimator_name == 'KMeans':
            self.metric_dataset = pd.DataFrame(metric_set, columns=['number of clusters', 'train dataset length', 'test dataset length',
                                                                     'silhouette metric', 'calinski metric', 'david bolduin metric'])
            self.metric_dataset.to_csv(f'metrics/kmeans_clustering.csv')

    def separate_clusters(self, dataset):
        for i in range(self._get_number_of_clusters(dataset)):
            cluster = dataset[dataset['clusters'] == i]
            self.clusters_list.append(cluster.drop('clusters', 1))

    def probability_per_cluster(self, dataset):
        total_length = len(dataset)
        for i in self.clusters_list:
            print(f'Cluster № {i+1} probability is {self._get_probability_per_cluster(total_length, len(i))} %')

    def best_clustering_find(self):
        best_metrics = list(
                                zip(
                                    ('silhouette metric', 'calinski metric', 'david bolduin metric'),
                                    (max(self.metric_dataset['silhouette metric']), max(self.metric_dataset['calinski metric']),
                                        min(self.metric_dataset['david bolduin metric']))
                                    )
                            )
        best_df = pd.DataFrame(columns=['number of clusters', 'train dataset length', 'test dataset length',
                                                                     'silhouette metric', 'calinski metric', 'david bolduin metric'])
        for ind, metric in best_metrics:
            print(self.metric_dataset[self.metric_dataset[ind] == metric])
            best_df.append(self.metric_dataset[self.metric_dataset[ind] == metric])
        best_df.to_csv('metrics/best_metrics.csv')

    def make_features_rate(self, importances, columns):
        pass

    def get_metric_dataset(self):
        return self.metric_dataset

    #   Closed methods
    def _get_number_of_clusters(self, dataset):
        return len(set(dataset['clusters']))

    def _get_probability_per_cluster(self, total_length, cluster_length):
        return (cluster_length/total_length) * 100

    @property
    def separated_clusters(self):
        return self.clusters_list
