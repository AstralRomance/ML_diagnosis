from sklearn.cluster import KMeans

'''
Class for estimators. Use downgraded data
'''


class Clustering:
    def __init__(self, dataset):
        self.data = dataset

#   KMeans estimator
    def kmeans_clustering(self, max_iter, random_state=0):
        classifier = KMeans(n_clusters=4, max_iter=max_iter, random_state=random_state).fit(self.data)
        predict_val = classifier.predict(self.data)
        return self.estimator_type(classifier), predict_val, classifier.cluster_centers_

#   Return estimator type as string
    def estimator_type(self, estimator):
        return str(type(estimator)).split('.')[-1][0:-2]

#   return downgraded dataset saved in clustering class
    def get_data(self):
        return self.data
