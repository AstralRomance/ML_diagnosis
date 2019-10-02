from itertools import takewhile, count

from sklearn.cluster import KMeans
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

import xls_to_csv_parser

class KMeans_make:
    def __init__(self, dataset):
        self.data = dataset

    def classification_start(self):
        for i in range(100, 5000, 200):
            classifier = KMeans(n_clusters=4, max_iter=i, random_state=0).fit(self.data)
            predict_val = classifier.predict(self.data)
            #print(classifier.cluster_centers_)
            self.make_plot(classifier, predict_val, classifier.cluster_centers_, i)

    def make_plot(self, classifier, pred_val, centers, max_it):
        fig = plt.figure()
        plt.scatter(self.data[:, 0], self.data[:, 1], c=pred_val, s=50, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        # plt.show()
        fig.savefig('graphs/kmeans/graph{0}.png'.format(max_it))
        plt.close(fig=fig)

