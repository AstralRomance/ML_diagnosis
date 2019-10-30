from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


'''
Classifier parent class
'''

class Classifier:
    def __init__(self, train, train_length):
        self.classifier = None
        self.classifier_data = train
        self.train_distr = self.make_train_distr(train_length)
        self.test_distr = self.make_test_distr(train_length)
        self.res_map = {}

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


class KMeansClassifier(Classifier):
    def __init__(self, data, cluster_num, train_length):
        super().__init__(train_length)
        self.classifier = None

    def training(self, clusters, max_iter, random_state=None):
        self.classifier = KMeans(n_clusters=clusters, max_iter=max_iter, random_state=random_state).fit(self.train_distr)

    def prediction(self):
        predict_val = self.classifier.predict(self.test_distr)
        for counter, cluster in enumerate(predict_val):
            self.res_map[cluster] = self.classifier_data[counter]
        return self.estimator_type(self.classifier)


class RegressionMethod(Classifier):
    def __init__(self, data, train_length):
        super().__init__(data, train_length)

    def training(self):
        self.classifier = LinearRegression()
        actual_train_data = self.train_distr.drop('К0011', 1)
        self.classifier.fit(actual_train_data, self._get_train_features())

    def prediction(self):
        actual_test_data = self.train_distr.drop('К0011', 1)
        return mean_absolute_error(self._get_test_true(), self.classifier.predict(actual_test_data)),\
               r2_score(self._get_test_true(), self.classifier.predict(actual_test_data))

    def _get_train_features(self):
        return self.train_distr['К0011'].values

    def _get_test_true(self):
        return self.train_distr['К0011'].values

