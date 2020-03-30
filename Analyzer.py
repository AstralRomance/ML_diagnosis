from scipy import stats
import pandas as pd
from collections import Counter

class Analyzer:
    def __init__(self, input_data=None):
        self.metric_dataset = None
        self.clusters_list = []

    def normal_check(self):
        '''
            Method check type of distribution in clusters
            Form dataframes with result of normal test and middle values per predictors
        :return: None
        '''
        cluster_counter = 0
        normal_check_list = []
        math_exp = []
        for cluster in self.clusters_list:
            pred_normal_test_res = []
            middle_val = []
            print(f'Current cluster nubmer {cluster_counter}')
            for predictor_list in cluster:
                print(f'Distribution for predictor: {predictor_list}')
                try:
                    if stats.normaltest(cluster[predictor_list])[1] < 3.27207e-11:
                        print('Seems normal')
                        pred_normal_test_res.append('ok')
                        middle_val.append(round(self.calc_avg_val(cluster[predictor_list]), 3))
                        print(f'Math expectation for predictor = {self.calc_avg_val(cluster[predictor_list])}')
                    else:
                        print('Cant rejected normal hypothesis')
                        pred_normal_test_res.append('nn')
                        middle_val.append(round(self.calc_avg_val(cluster[predictor_list]), 3))
                except Exception as e:
                    print(f'Exception {e} raised with normaltest')
                    pred_normal_test_res.append('-1')
                    middle_val.append(-1)
            normal_check_list.append(pred_normal_test_res)
            math_exp.append(middle_val)
            cluster_counter += 1
        normal_df = pd.DataFrame(normal_check_list, columns=[i for i in self.clusters_list[0]])
        normal_df.to_csv('metrics/normal_check.csv')
        math_exp_per_predictor = pd.DataFrame(math_exp, columns=[i for i in self.clusters_list[0]])
        math_exp_per_predictor.to_csv('metrics/avg_values.csv')

    def metric_collection(self, estimator_name, metric_set):
        '''
            Makes csv for Kmeans mertics
        :param estimator_name:
        :param metric_set: list of metrics and length parameter
        :return: None
        '''
        if estimator_name == 'KMeans':
            self.metric_dataset = pd.DataFrame(metric_set, columns=['number of clusters', 'train dataset length', 'test dataset length',
                                                                     'silhouette metric', 'calinski metric', 'david bolduin metric'])
            self.metric_dataset.to_csv(f'metrics/kmeans_clustering.csv')

    def separate_clusters(self, dataset):
        '''
            Form list with separated clusters
        :param dataset: clustered dataset
        :return: None
        '''
        for i in range(self._get_number_of_clusters(dataset)):
            cluster = dataset[dataset['clusters'] == i]
            self.clusters_list.append(cluster.drop('clusters', 1))

    def probability_per_cluster(self, dataset):
        '''
            Count probability of entering in the cluster
        :param dataset: clustered dataframe
        :return: None
        '''
        total_length = len(dataset)
        counter = 1
        for i in self.clusters_list:
            print(f'Cluster № {counter} probability is {self._get_probability_per_cluster(total_length, len(i))} %')
            counter += 1

    def best_clustering_find(self):
        '''
            Find and form dataframe with best metrics of clustering.
        :return: None
        '''
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
        '''
        :param forest_features: values of random forest features
        :param cols: columns of dataset
        :param train_l: length of train distribution. Use as part of pathfile
        :return:
        '''
        features = pd.DataFrame({'feature': cols, 'importance': forest_features}).sort_values(
            'importance', ascending=False)
        features.to_csv(f'metrics/random_forest_weights/predictors_weights_train_l_{train_l}.csv')

    def calc_avg_val(self, predictors_list):
        '''
        :param predictors_list: list of predictor values
        :return: math expectation for normal distribution
        '''
        return sum(predictors_list)/len(predictors_list)

    def get_metric_dataset(self):
        return self.metric_dataset

    def calc_predictors_interval(self):
        predictor_scopes = None
        for counter, cluster in enumerate(self.clusters_list):
            print(f'calculating intervals for cluster {counter}')
            predictor_scopes = pd.DataFrame(
                                            [
                                                [min(cluster[cluster[column] >= 0][column]) for column in cluster],
                                                [max(cluster[cluster[column] >= 0][column]) for column in cluster],
                                                list(map(lambda x, y: x-y, [max(cluster[cluster[column] >= 0][column]) for column in cluster], [min(cluster[cluster[column] >= 0][column]) for column in cluster]))
                                            ],
                                            columns=[column for column in cluster],
                                            index=['min', 'max', 'range']
                                            )
            predictor_scopes.T.to_csv(f'metrics/range_of_predictor_cluster{counter}.csv')

    def make_probs_for_pred(self, predictor):
        cnt = Counter(predictor)
        return {i: cnt[i]/len(predictor) for i in cnt.keys()}

    #Closed methods
    def _get_number_of_clusters(self, dataset):
        '''
        :param dataset: dataframe with clusters
        :return: number of clusters
        '''
        return len(set(dataset['clusters']))

    def _get_probability_per_cluster(self, total_length, cluster_length):
        '''
        :param total_length: length of clustered data
        :param cluster_length: length of cluster
        :return: probability of entering in cluster
        '''
        return (cluster_length/total_length) * 100

    @property
    def separated_clusters(self):
        '''
        :return: separated clusters
        '''
        return self.clusters_list

