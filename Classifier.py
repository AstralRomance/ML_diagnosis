import os
from abc import ABC, abstractmethod

from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from sklearn.model_selection import cross_val_score

'''
Classifier parent class
'''

class Classifier:
    def __init__(self, data, train_length):
        '''
        :param data: dataset to estimator after parsing
        :param train_length: train distribution length
        '''
        self.classifier = None
        self.res_map = None
        self.classifier_data = data.fillna(-1)
        self.train_distr = self.make_train_distr(train_length)
        self.test_distr = self.make_test_distr(train_length)

    '''
        Estimators abstract methods for training, prediction and metric collection
    '''
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def metric_collection(self):
        raise NotImplementedError

    def make_train_distr(self, train_length):
        '''
        :param train_length: length of train distribution
        :return: train distribution for estimator
        '''
        return self.classifier_data[0:train_length]

    def make_test_distr(self, train_length):
        '''
        :param train_length: length of train distribution
        :return: test distribution for estimator
        '''
        return self.classifier_data[train_length::]

    @property
    def get_data(self):
        '''
        :return: test distribution
        '''
        return self.test_distr

    @property
    def estimator_type(self):
        '''
        :return: estimator type as string
        '''
        return str(type(self.classifier)).split('.')[-1][0:-2]

    @property
    def downgraded_dataset(self):
        '''
        Not used
        '''
        return self.classifier_data

    @property
    def get_test(self):
        '''
        :return: test distribution dataframe
        '''
        return self.test_distr

    @property
    def data_length(self):
        '''
        :return: length of all dataset
        '''
        return len(self.classifier_data)


class KMeansClassifier(Classifier):
    def __init__(self, data, train_length):
        super().__init__(data, train_length)
        self.dataset_clustered_labels = None

    def train(self, clusters=3, max_iter=300):
        '''
            Train kmeans estimator with data used in __init__ method
        :param clusters: number of clusters
        :param max_iter: maximum of kmeans iterations
        :return: None
        '''
        self.classifier = KMeans(n_clusters=clusters, max_iter=max_iter)
        self.classifier.fit(self.train_distr)

    def predict(self):
        '''
            Test for kmeans estimator
        :return: None
        '''
        prediction = self.classifier.predict(self.get_test)
        self.dataset_clustered_labels = self.test_distr.copy()
        self.dataset_clustered_labels['clusters'] = prediction

    def clusters_to_excel(self):
        self.dataset_clustered_labels.to_excel('clustered_dataframe.xlsx', sheet_name='Sh1')

    def _kmeans_silhouette(self):
        '''
        :return: silhouette metric. -10 if raised ValueError
        '''
        try:
            return silhouette_score(self.test_distr, self.dataset_clustered_labels['clusters'])
        except ValueError as val_err:
            print(f'found {val_err} in {len(self.train_distr)} and {len(self.train_distr)}')
            return -10

    def _calinski(self):
        '''
        :return: calinski metric.
        '''
        try:
            return calinski_harabasz_score(self.test_distr, self.dataset_clustered_labels['clusters'])
        except ValueError as val_err:
            print(f'found {val_err} in {len(self.train_distr)} and {len(self.train_distr)}')
            return -10

    def _d_b_score(self):
        '''
        :return: davies bolduin metric
        '''
        try:
            return davies_bouldin_score(self.test_distr, self.dataset_clustered_labels['clusters'])
        except ValueError as val_err:
            print(f'found {val_err} in {len(self.train_distr)} and {len(self.train_distr)}')
            return -10

    @property
    def get_labels(self):
        '''
        :return: labels of clustered dataframe
        '''
        return self.dataset_clustered_labels['clusters']

    @property
    def get_clustered(self):
        '''
        :return: datagrame with clusters
        '''
        return self.dataset_clustered_labels

    @property
    def metrics(self):
        '''
        :return: tuple of metrics
        '''
        return len(set(self.classifier.labels_)), len(self.train_distr), len(self.test_distr), self._kmeans_silhouette(), self._calinski(), self._d_b_score()


class BayesClassifier(Classifier):
    '''
    This thing is not use now
    '''
    def __init__(self, data, train_length, class_label):
        super().__init__(data, train_length)
        self.prediction = None
        self.class_label = class_label

    def train(self):
        self.train_distr = self.train_distr.fillna(-1)
        self.classifier = GaussianNB()
        self.classifier.fit(self.train_distr.drop(self.class_label, 1), self.train_distr[self.class_label])
        return self.classifier.score(self.train_distr.drop(self.class_label, 1), self.train_distr[self.class_label])

    def predict(self):
        self.test_distr = self.test_distr.fillna(-1)
        self.prediction = self.classifier.predict(self.test_distr.drop(self.class_label, 1))
        return self.classifier.score(self.test_distr.drop(self.class_label, 1), self.test_distr[self.class_label])

    @property
    def get_prediction(self):
        return self.prediction

class Forest(Classifier):
    def __init__(self, data, class_label, train_length):
        '''
        :param data: dataset after parsing
        :param class_label: column with class feature
        :param train_length: train distribution length
        '''
        data = data.fillna(-1)
        super().__init__(data, train_length)
        self.class_label = class_label

    def train(self, max_depth=2, random_state=0):
        '''
            Random forest training
        :param max_depth: random forest max length
        :param random_state:
        :return: None
        '''
        self.classifier = RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=101)
        self.classifier.fit(self.train_distr.drop(self.class_label, 1), self.train_distr[self.class_label])

    def predict(self):
        '''
            Random forest class prediction
        :return: random forest prediction
        '''
        pred = self.classifier.predict(self.test_distr.drop(self.class_label, 1))
        return pred

    def collect_test_score(self, mode='score'):
        '''
        :param mode:
        :return:
        '''
        if mode == 'score':
            self.score_write(
                self.classifier.score(self.test_distr.drop(self.class_label, 1), self.test_distr[self.class_label]),
                'test', len(self.train_distr))
            return self.classifier.score(self.test_distr.drop(self.class_label, 1), self.test_distr[self.class_label])
        elif mode == 'cross_validate':
            '''
            self.score_write(
                cross_val_score(self.test_distr.drop(self.test_distr.drop(self.class_label, 1)), self.test_distr[self.class_label], cv=4),
                                'test', len(self.train_distr)
            )
            '''
            return cross_val_score(self.classifier, self.test_distr.drop(self.class_label, 1), self.test_distr[self.class_label], cv=4)

    def collect_train_score(self, mode='score'):
        '''
        :return: score metric on train distribution
        '''
        if mode == 'score':
            self.score_write(self.classifier.score(self.train_distr.drop(self.class_label, 1), self.train_distr[self.class_label]),
                             'train', len(self.train_distr))
            return self.classifier.score(self.train_distr.drop(self.class_label, 1), self.train_distr[self.class_label])
        elif mode == 'cross_validate':
            '''
            self.score_write(
                cross_val_score(self.train_distr.drop(self.train_distr.drop(self.class_label, 1), self.train_distr[self.class_label]), cv=4
                                ), 'train', len(self.train_distr)
            )
            '''
            return cross_val_score(self.classifier, self.train_distr.drop(self.class_label, 1), self.train_distr[self.class_label], cv=4)

    def score_write(self, score, distr, train_len):
        '''
            method write scores into file
        :param score: score value
        :param distr: type of distribution (train or test) use as part of filepath
        :param train_len: length of used distribution use as part of filepath
        :return: None
        '''
        with open(f'metrics/random_forest_{distr}/forest_score_{train_len}.txt', 'w') as outp:
            outp.write(str(round(score, 5)))

    @property
    def get_feature_importances(self):
        '''
        :return: feature importances values
        '''
        return self.classifier.feature_importances_
