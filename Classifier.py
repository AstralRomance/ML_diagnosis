from abc import ABC, abstractmethod

from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

'''
Classifier parent class
'''

class Classifier:
    def __init__(self, data, train_length):
        self.classifier = None
        self.res_map = None
        self.classifier_data = data.fillna(-1)
        self.train_distr = self.make_train_distr(train_length)
        self.test_distr = self.make_test_distr(train_length)

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
        return self.classifier_data[0:train_length]

    def make_test_distr(self, train_length):
        return self.classifier_data[train_length::]

    @property
    def estimator_type(self):
        return str(type(self.classifier)).split('.')[-1][0:-2]

    @property
    def downgraded_dataset(self):
        return self.classifier_data

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

    def _kmeans_silhouette(self):
        try:
            return silhouette_score(self.test_distr, self.dataset_clustered_labels['clusters'])
        except ValueError as val_err:
            print(f'found {val_err} in {len(self.train_distr)} and {len(self.train_distr)}')
            return -10

    def _calinski(self):
        try:
            return calinski_harabasz_score(self.test_distr, self.dataset_clustered_labels['clusters'])
        except ValueError as val_err:
            print(f'found {val_err} in {len(self.train_distr)} and {len(self.train_distr)}')
            return -10

    def _d_b_score(self):
        try:
            return davies_bouldin_score(self.test_distr, self.dataset_clustered_labels['clusters'])
        except ValueError as val_err:
            print(f'found {val_err} in {len(self.train_distr)} and {len(self.train_distr)}')
            return -10

    @property
    def get_clustered(self):
        return self.dataset_clustered_labels

    @property
    def metrics(self):
        return len(set(self.classifier.labels_)), len(self.train_distr), len(self.test_distr), self._kmeans_silhouette(), self._calinski(), self._d_b_score()


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

    @property
    def get_test_labels(self):
        return self.test_distr[self.class_label]

class Forest(Classifier):
    def __init__(self, data, train_length, class_label):
        data = data.fillna(-1)
        super().__init__(data, train_length)
        self.class_label = class_label

    def train(self, max_depth=2, random_state=0):
        self.classifier = RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=101)
        self.classifier.fit(self.train_distr.drop(self.class_label, 1), self.train_distr[self.class_label])
        return self.classifier.feature_importances_
        #return self.classifier.score(self.train_distr.drop(self.class_label, 1), self.train_distr[self.class_label])
        #return {self.train_distr.keys():self.classifier.feature_importances_}

    def predict(self):
        pred = self.classifier.predict([self.test_distr[self.class_label]])
        print(self.classifier.score(pred, self.test_distr[self.class_label]))
        return self.classifier.score(pred, self.test_distr[self.class_label])