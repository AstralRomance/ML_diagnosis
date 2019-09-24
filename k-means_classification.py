from sklearn.cluster import KMeans
import matplotlib
from matplotlib import pyplot
import pandas as pd
from sklearn import metrics

import xls_to_csv_parser


def make_plot(my_data, classifier):
    colormap = matplotlib.pyplot.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=40)
    axes = pd.plotting.scatter_matrix(my_data, color=colormap(norm(classifier.labels_)))


def KMeans_cluster():
    data = xls_to_csv_parser.parser()
    for i in range(30):
        kmeans_classifier = KMeans(n_clusters=3, max_iter=150).fit(data)
        with open('prdc.txt', 'a') as tr:
            tr.write(str(kmeans_classifier.fit_transform(data)))
            tr.write('\n\n')

        with open('res.txt', 'a') as outp:
            outp.write(str(kmeans_classifier.cluster_centers_))
            outp.write('\n\n')






KMeans_cluster()
#matplotlib.pyplot.show()
