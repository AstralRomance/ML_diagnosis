from sklearn.cluster import KMeans
import matplotlib
from matplotlib import pyplot
import pandas as pd

import xls_to_csv_parser

class KmeansTest:
    def __init__(self):
        prs = xls_to_csv_parser.Parser()
        self.data = prs.parser()

    def make_plot(self, classifier):
        colormap = matplotlib.pyplot.cm.rainbow
        norm = matplotlib.colors.Normalize(vmin=0, vmax=40)
        axes = pd.plotting.scatter_matrix(self.data, color=colormap(norm(classifier.labels_)))

    def KMeans_cluster(self):
        for i in range(30):
            kmeans_classifier = KMeans(n_clusters=3, max_iter=150).fit(self.data)
            with open('prdc.txt', 'a') as tr:
                tr.write(str(kmeans_classifier.fit_transform(self.data)))
                tr.write('\n\n')
            with open('res.txt', 'a') as outp:
                outp.write(str(kmeans_classifier.cluster_centers_))
                outp.write('\n\n')
            self.make_plot(kmeans_classifier)
            pyplot.show()