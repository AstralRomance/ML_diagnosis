from Parser import Parser
from Downgrader import Downgrader
from Clustering import Clustering
from Visualizer import Visualizer
from Analyzer import Analyzer

parser = Parser('datasetxls.xlsx')
plotter = Visualizer()
# Dataset taken
downgrader = Downgrader(parser.parse())
for i in range(50, 1000, 50):
    for j in range(10, 100, 10):
        for k in range(100, 5000, 200):
            kmeans_clusters = Clustering(downgrader.downgrade(i, j))
            res_cl = kmeans_clusters.kmeans_clustering(k)
            params = (res_cl[0], kmeans_clusters.get_data,
                      'tsne_parameters_' + str(i) + '_' + str(j) + '_kmeans_parameters_' + str(k), res_cl[1], res_cl[2])
            plotter.make_euclidean_space_plot(*params)
            analyzer = Analyzer(res_cl[1])
            analyzer.probability_calc()
