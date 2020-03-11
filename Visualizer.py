from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Visualizer:
    def make_euclidean_scatter_plot(self, estimator_name, data, param, predicted=None, centers=None):
        '''
            Form a scatter graph.
        :param estimator_name: part of path. Using as directory name
        :param data: points coordinates to make graph
        :param param: estimator parameters. Use as graph name
        :param predicted: use as color sequence
        :param centers: centers of clusters
        :return: None
        '''
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

    def make_compare_plot(self, points_x, points_y, name):
        '''
            Form graphs from lists in points_y.
        :param points_x: points in ox. Same for all lists from points_y
        :param points_y: list of point lists.
        :param name: name of graph
        :return: None
        '''
        fig = plt.figure()
        number_of_graphics = max(list(map(len, [i for i in points_y])))
        gs = gridspec.GridSpec(number_of_graphics, 1, fig)
        for i in range(number_of_graphics):
            fig.add_subplot(gs[i, 0], )
            plt.plot(points_x, [j[i] for j in points_y])
        plt.savefig(f'graphs/{name}.png')

    def make_simple_graph(self, points_x, i):
        '''
            Form a simple graph
        :param points_x: values for graph
        :param i: parameter use as number for graph name
        :return: None
        '''
        fig = plt.figure()
        plt.plot([x for x in points_x])
        plt.savefig(f'graphs/normal_analysis/intervals with step {i}.png')
        plt.close(fig)

    def make_heatmap(self, data, ages=None,  heatmap_name = None, cluster_n = None):
        '''
            Form a heatmap
        :param data: data for heatmap
        :param ages: ages columns for drop
        :return: None
        '''
        plt.figure(figsize=(20,20))
        try:
            for age in ages:
                data = data.drop(str(age), 1)
        except Exception as e:
            print(f'Found {e} while heatmap forming')
        sns.heatmap(data.corr(), annot=True, linewidths=.3, cbar=False, square=True)
        plt.savefig(f'graphs/clasification_heatmap/{heatmap_name}_{cluster_n}.png')
        plt.close('all')

    def make_pairplot(self, data, ages, name='new_pairplot'):
        '''
            Form a plot in features space
        :param data: data for graph
        :param ages: columns for drop
        :param name: string use as path of filename
        :return: None
        '''
        if str(ages[0]) in data.columns:
            for age in ages:
                data = data.drop(str(age), 1)
        sns.pairplot(data, hue='clusters', diag_kind='hist')
        plt.savefig(f'graphs/pairplots/pairplot_{name}.png')
        plt.close('all')

    def make_overlearn_check_plot(self, train_sc, test_sc, ranging, clname):
        '''
            Form a plot for train and test distributions.
        :param train_sc: points in train distribution
        :param test_sc: points in test distribution
        :param ranging: number of points. Same for train and test lists.
        :param clname: string use as path of filename
        :return: None
        '''
        plt.plot(ranging, train_sc, label='train distr')
        plt.plot(ranging, test_sc, label='test distr')
        plt.legend()
        plt.savefig(f'graphs/{clname}.png')
        plt.close('all')

    def make_confusion_matrix(self, true_labels, predicted_labels, train_l):
        '''
            Form a confusion matrix.
        :param true_labels: true labels from dataset
        :param predicted_labels: predicted labels from dataset
        :param train_l: length of train distribution. Use as part of filename
        :return: None
        '''
        plt.figure(figsize=(15, 15))
        confusion = confusion_matrix(true_labels, predicted_labels)
        sns.heatmap(confusion, annot=True, square=True)
        plt.savefig(f'graphs/classification_heatmap/heatmap_{train_l}.png', bbox_inches='tight')
        with open(f'heatmaps/heatmap{train_l}.txt', 'w') as hmp:
            hmp.write(str(confusion))
        plt.close('all')

    def distribution_hist(self, predictor, cluster_n, predictor_name):
        '''
            Form a histogram.
        :param predictor: list of predictor values
        :param cluster_n: cluster number
        :param predictor_name: name of predictor
        :return:
        '''
        sorted_features = sorted(predictor)
        sns.kdeplot(sorted_features)
        plt.savefig(f'graphs/predictor_per_cluster_distribution/distribution_in_{cluster_n}_for_{predictor_name}.png')
        plt.close('all')
