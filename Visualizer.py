from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
import seaborn as sns

'''
Class for make plots
'''


class Visualizer:
#   Make plot in euclidean space
    def make_euclidean_space_plot(self, estimator_name, data, param, predicted=None, centers=None):
        fig = plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=predicted, s=50, cmap='viridis')
        if not centers.all():
            fig.savefig(f'graphs/{estimator_name}/{param}')
            plt.close(fig)
            return
        else:
            plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
            fig.savefig(f'graphs/{estimator_name}/{param}')
            plt.close(fig)

#   Plot for probability change visualising
    def make_probability_plot(self, probability):
        plt.plot([i for i in probability[0]])
        plt.plot([i for i in probability[1]])
        plt.plot([i for i in probability[2]])
        plt.plot([i for i in probability[3]])
        plt.savefig('prob_change.png')

    def make_compare_plot(self, points_x, points_y, name):
        fig = plt.figure()
        number_of_graphics = max(list(map(len, [i for i in points_y])))
        gs = gridspec.GridSpec(number_of_graphics, 1, fig)
        for i in range(number_of_graphics):
            fig.add_subplot(gs[i, 0], )
            plt.plot(points_x, [j[i] for j in points_y])
        plt.savefig(f'{name}.png')

    def make_simple_graph(self, points_x, i):
        fig = plt.figure()
        plt.plot([x for x in points_x])
        plt.savefig(f'graphs/normal_analysis/intervals with step {i}.png')
        plt.close(fig)

    def make_heatmap(self, data, ages):
        for age in ages:
            data = data.drop(str(age), 1)
        sns.heatmap(data.corr(), annot=True)
        plt.savefig(f'graphs/heatmap.png')
        plt.show()

    def make_pairplot(self, data, ages, name='new_pairplot'):
        for age in ages:
            data = data.drop(str(age), 1)
        g = sns.pairplot(data, hue='clusters', diag_kind='hist')
        plt.savefig(f'graphs/pairplots/pairplot_{name}.png')
        plt.close('all')

    def make_overlearn_check_plot(self, train_sc, test_sc, ranging, clname):
        plt.plot(ranging, train_sc, label='train distr')
        plt.plot(ranging, test_sc, label='test distr')
        plt.legend()
        plt.savefig(f'graphs/{clname}.png')
        plt.close('all')

    def make_cluster_hist(self, clusters):
        plt.hist(clusters)
        plt.show()

    def make_confusion_matrix(self, true_labels, predicted_labels, train_l):
        fig = plt.figure(figsize=(9, 12))
        confusion = confusion_matrix(true_labels, predicted_labels)
        sns.heatmap(confusion, annot=True, square=True)
        plt.savefig(f'graphs/classification_heatmap/heatmap_{train_l}.png', bbox_inches='tight')
        with open(f'heatmaps/heatmap{train_l}.txt', 'w') as hmp:
            hmp.write(str(confusion))
        plt.close(fig)
