import Classifier
from Parser import Parser
from Downgrader import Downgrader
from Visualizer import Visualizer
from Analyzer import Analyzer

parser = Parser('datasetxls.xlsx')
plotter = Visualizer()
# Dataset taken
downgrader = Downgrader(parser.parse())
analyzer = Analyzer()
#for i in range(50, 1000, 50):
#    for j in range(10, 100, 10):
#        pass
"""
        DEBUG VERSION. BLOCK FOR KMEANS CLUSTERING USAGE
        for k in range(100, 5000, 200):
            
            kmeans_clusters = Classifier(downgrader.downgrade(i, j))
            res_cl = kmeans_clusters.kmeans_clustering(k)
            params = (res_cl[0], kmeans_clusters.get_data,
                      'tsne_parameters_' + str(i) + '_' + str(j) + '_kmeans_parameters_' + str(k), res_cl[1], res_cl[2])
            plotter.make_euclidean_space_plot(*params)
            analyzer.probability_calc(kmeans_clusters.get_clustering_map)
"""
#THE BEST TRAINING LENGTH = 1150
abs_err = []
for i in range(200, 1500, 20):
    regression = Classifier.RegressionMethod(downgrader.get_unmodified_data, i)
    regression.training()
    abs_err.append(regression.prediction())

plotter.make_linear_plot([i for i in range(200, 1500, 20)], abs_err)
#plotter.make_probability_plot(analyzer.get_total_probability)
