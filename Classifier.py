from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


'''
Classifier parent class
'''


class Classifier:
    def __init__(self, train):
        self.classifier_data = train
        self.res_map = {}

    def estimator_type(self, estimator):
        return str(type(estimator)).split('.')[-1][0:-2]

    @property
    def downgraded_dataset(self):
        return self.classifier_data

    @property
    def get_result_map(self):
        return self.res_map


class KMeansClassifier(Classifier):
    def training(self, max_iter, random_state=None):
        classifier = KMeans(n_clusters=10, max_iter=max_iter, random_state=random_state).fit(self.classifier_data)
        predict_val = classifier.predict(self.classifier_data)
        for counter, cluster in enumerate(predict_val):
            self.res_map[cluster] = self.classifier_data[counter]
        return self.estimator_type(classifier)

    def prediction(self):
        pass


class RegressionMethod(Classifier):
    def __init__(self, train, train_length):
        super().__init__(train)
        self.classifier = None
        self.train_length = train_length

    def training(self):
        self.classifier = LinearRegression()
        self.classifier.fit(self._get_train_data(), self._get_train_features())

    def prediction(self):
        t = []
        for i in self._get_test():
            t.append(*self.classifier.predict([i]))
        return (mean_absolute_error(self._get_test_true(), t), r2_score(self._get_test_true(), t))

    def drop_parameters(self):
        pass

    def _get_train_data(self, drops=None):
        return self.classifier_data[0:self.train_length:].drop('К0011', 1).values

    def _get_train_features(self):
        return self.classifier_data['К0011'].values[0:self.train_length:]

    def _get_test(self, drops=None):
        return self.classifier_data[self.train_length::].drop('К0011', 1).values

    def _get_test_true(self):
        return self.classifier_data['К0011'].values[self.train_length::]


