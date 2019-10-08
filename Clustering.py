from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

'''
Class for estimators. Use downgraded data
'''


class Clustering:
    def __init__(self, train, test=None):
        self.train_data = train
        if not test:
            self.test_data = self.train_data
        else:
            self.test_data = test
        self.res_map = {}

#   KMeans estimator
    def kmeans_clustering(self, max_iter, random_state=0):
        classifier = KMeans(n_clusters=4, max_iter=max_iter, random_state=random_state).fit(self.train_data)
        predict_val = classifier.predict(self.test_data)
        temp = [[], [], [], []]
        for i, j in enumerate(predict_val):
            temp[j].append(self.train_data[i])
        for i, j in enumerate(temp):
            self.res_map[i] = j
        print(self.res_map)
        return self.estimator_type(classifier), predict_val, classifier.cluster_centers_

#   Return estimator type as string
    def estimator_type(self, estimator):
        return str(type(estimator)).split('.')[-1][0:-2]

#   return downgraded dataset saved in clustering class
    @property
    def get_data(self):
        return self.test_data

    @property
    def get_clustering_map(self):
        return self.res_map