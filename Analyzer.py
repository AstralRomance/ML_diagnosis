from scipy import stats
import pandas as pd


class Analyzer:
    def __init__(self, input_data=None):
        self.metric_dataset = None
        self.clusters_list = []

    def normal_check(self):
        cluster_counter = 0
        normal_check_list = []
        math_exp = []
        for cluster in self.clusters_list:
            temp = []
            temp1 = []
            print(f'Current cluster nubmer {cluster_counter}')
            for predictor_list in cluster:
                print(f'Current distribution for predictor: {predictor_list}')
                try:
                    if stats.normaltest(cluster[predictor_list])[1] < 3.27207e-11:
                        print('Seems normal')
                        temp.append('ok')
                        print(f'Math expectation for predictor = {self.math_expectation(cluster[predictor_list])}')
                        temp1.append(round(self.math_expectation(cluster[predictor_list]), 3))
                    else:
                        print('Cant rejected normal hypothesis')
                        temp.append('nn')
                        temp1.append('nn')
                except Exception as e:
                    print(f'Exception {e} raised with normaltest')
                    temp.append('-1')
                    temp1.append(-1)
            normal_check_list.append(temp)
            math_exp.append(temp1)
            cluster_counter += 1
        normal_df = pd.DataFrame(normal_check_list, columns=[i for i in self.clusters_list[0]])
        normal_df.to_csv('metrics/normal_check.csv')
        math_exp_per_predictor = pd.DataFrame(math_exp, columns=[i for i in self.clusters_list[0]])
        math_exp_per_predictor.to_csv('metrics/math_expectations.csv', sep=';')

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
        counter = 1
        for i in self.clusters_list:
            print(f'Cluster № {counter} probability is {self._get_probability_per_cluster(total_length, len(i))} %')
            counter += 1

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

    def make_features_rate(self, forest_features, cols, train_l):
        features = pd.DataFrame({'feature': cols, 'importance': forest_features}).sort_values(
            'importance', ascending=False)
        features.to_csv(f'metrics/random_forest_weights/predictors_weights_train_l_{train_l}.csv')

    def get_metric_dataset(self):
        return self.metric_dataset

    #   Closed methods
    def _get_number_of_clusters(self, dataset):
        return len(set(dataset['clusters']))

    def _get_probability_per_cluster(self, total_length, cluster_length):
        return (cluster_length/total_length) * 100

    def math_expectation(self, predictors_list):
        return sum(predictors_list)/len(predictors_list)

    @property
    def separated_clusters(self):
        return self.clusters_list
