from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

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
    def training(self, max_iter, random_state):
        classifier = KMeans(n_clusters=4, max_iter=max_iter, random_state=random_state).fit(self.classifier_data)
        predict_val = classifier.predict(self.classifier_data)
        temp = [[], [], [], []]
        for i, j in enumerate(predict_val):
            temp[j].append(self.classifier_data[i])
        for i, j in enumerate(temp):
            self.res_map[i] = j
        print(self.res_map)
        return self.estimator_type(classifier), self.res_map

    def prediction(self):
        pass


class RegressionMethod(Classifier):
    def __init__(self, train):
        super().__init__(train)
        self.classifier = None

    def training(self):
        self.classifier = LinearRegression()
        self.classifier.fit(self._get_train_data(), self._get_train_features())

    def prediction(self):
        for i in self._get_test():
            t = self.classifier.predict([i])
            print(t)

    def _get_train_data(self):
        return self.classifier_data[0:1000:].drop('К0011', 1).values

    def _get_train_features(self):
        return self.classifier_data['К0011'].values[0:1000:]

    def _get_test(self):
        return self.classifier_data[1000::].drop('К0011', 1).values

