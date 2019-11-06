from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
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


    def make_heatmap(self, data):
        sns.heatmap(data.corr(), annot=True)
        plt.show()
