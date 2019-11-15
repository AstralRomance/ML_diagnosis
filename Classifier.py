from abc import ABC, abstractmethod
from sklearn. preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

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
    def __init__(self, data, train_length=100):
        super().__init__(data, train_length)
        self.clusters = []

    def train(self, clusters=3, max_iter=300):
        self.classifier = KMeans(n_clusters=clusters, max_iter=max_iter)
        self.classifier.fit(self.train_distr)

    def predict(self):
        prediction = self.classifier.predict(self.test_distr)
        self.clusters = prediction
        return prediction

    def choose_clustering_columns(self, valid_columns):
        for col in self.train_distr:
            if col not in valid_columns:
                self.train_distr.drop(col, 1)
                self.test_distr.drop(col, 1)

    def make_resulting_dataset(self):
        cl = [f'cluster {i}' for i in self.clusters]
        self.test_distr['clusters'] = cl
        #temp = self.train_distr
        #temp['clusters'] = cl
        return self.test_distr



# NON VALID CLASS
class RegressionMethod(Classifier):
    def __init__(self, data, train_length):
        super().__init__(data, train_length)

    def train(self):
        self.classifier = LinearRegression()
        actual_train_data = self.train_distr.drop('К0011', 1)
        self.classifier.fit(actual_train_data, self._get_train_features())

    def predict(self):
        actual_test_data = self.train_distr.drop('К0011', 1)
        return mean_absolute_error(self._get_test_true(), self.classifier.predict(actual_test_data)),\
               r2_score(self._get_test_true(), self.classifier.predict(actual_test_data))

    def _get_train_features(self):
        return self.train_distr['К0011'].values

    def _get_test_true(self):
        return self.train_distr['К0011'].values

