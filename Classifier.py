from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


'''
Classifier parent class
'''

class Classifier:
    def __init__(self, data, train_length):
        self.classifier = None
        self.res_map = None
        self.classifier_data = data
        self.train_distr = self.make_train_distr(train_length)
        self.test_distr = self.make_test_distr(train_length)

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    def estimator_type(self, estimator):
        return str(type(estimator)).split('.')[-1][0:-2]

    def make_train_distr(self, train_length):
        return self.classifier_data[0:train_length]

    def make_test_distr(self, train_length):
        return self.classifier_data[train_length::]

    @property
    def downgraded_dataset(self):
        return self.classifier_data

    @property
    def get_result_map(self):
        return self.res_map

    @property
    def get_test(self):
        return self.test_distr


class KMeansClassifier(Classifier):
    def __init__(self, data, train_length):
        super().__init__(data, train_length)
        self.dataset_clustered_labels = None

    def train(self, clusters=3, max_iter=300):
        self.classifier = KMeans(n_clusters=clusters, max_iter=max_iter)
        self.classifier.fit(self.train_distr)

    def predict(self):
        prediction = self.classifier.predict(self.get_test)
        self.dataset_clustered_labels = self.test_distr.copy()
        self.dataset_clustered_labels['clusters'] = prediction

    def choose_clustering_columns(self, valid_columns):
        for col in self.train_distr:
            if col not in valid_columns:
                self.train_distr.drop(col, 1)
                self.test_distr.drop(col, 1)

    @property
    def get_clustered(self):
        return self.dataset_clustered_labels


class BayesClassifier(Classifier):
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
