from matplotlib import pyplot as plt

'''
Class for make plots
'''


class Visualizer:
#   Make plot in euclidean space.
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