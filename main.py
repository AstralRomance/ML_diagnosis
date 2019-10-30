import time

import numpy as np
from matplotlib import pyplot as plt


import Classifier
from Parser import Parser
from Downgrader import Downgrader
from Visualizer import Visualizer
from Analyzer import Analyzer

parser = Parser('datasetxls.xlsx')
plotter = Visualizer()
# Dataset taken
downgrader = Downgrader(parser.parse())
analyzer = Analyzer(downgrader.get_unmodified_data)
'''
for i in range(50, 1000, 50):
    for j in range(10, 100, 10):
#        DEBUG VERSION. BLOCK FOR KMEANS CLUSTERING USAGE
        for k in range(100, 5000, 200):
            kmeans_clusters = Classifier.KMeansClassifier(downgrader.downgrade(i, j))
            res_cl = kmeans_clusters.training(k)
            params = ()
            plotter.make_euclidean_space_plot(*params)
            print(analyzer.probability_per_cluster(kmeans_clusters.get_result_map))
'''
#THE BEST TRAINING LENGTH = 1150
#abs_err = []
#for i in range(200, 1900, 10):
#    regression = Classifier.RegressionMethod(downgrader.get_unmodified_data, i)
#    regression.training()
#    abs_err.append(regression.prediction())

#plotter.make_compare_plot([i for i in range(200, 1900, 10)], abs_err, 'abs_error_and_r2_comparison')

print((downgrader.get_unmodified_data['К0011'].values))
print(analyzer.normal_check(downgrader.get_unmodified_data['К0011'].values))
print(analyzer.normal_check(downgrader.get_unmodified_data['К0011'].values))
for i in np.arange(0.1, 1.1, 0.1):
    intervals = analyzer.make_intervals(i)
    for interval in intervals.values():
        try:
            print(f'current step is {i} {analyzer.normal_check(interval)}, interval length {len(interval)}')
            print(f'current step is {i} {analyzer.normal_check(interval)}')
        except ValueError:
            print(f'current step is {i} and something goes wrong, current interval length {len(interval)}')

regression_error = []
for i in range(200, 1900, 10):
    regression = Classifier.RegressionMethod(downgrader.get_unmodified_data, i)
    regression.training()
    regression_error.append(regression.prediction())
plotter.make_compare_plot([i for i in range(200, 1900, 10)], regression_error, 'abs_error_and_r2_comparison')

