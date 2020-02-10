import os

class Setup:
    def __init__(self):
        self.dir_list = os.listdir(os.getcwd())
        self._check_metrics_dir()
        self._check_graphs_dir()
        self._check_sources_dir()

    def _check_sources_dir(self):
        if 'sources' not in self.dir_list:
            os.mkdir('sources')
            print('sources dir setup succesfully')
        else:
            print('sources dir already setup')

    def _check_metrics_dir(self):
        if 'metrics' not in self.dir_list:
            os.mkdir('metrics')
            print('metrics dir setup succesfully')
            self._check_submetrics_dir()
        else:
            print('metrics dir already setup')

    def _check_submetrics_dir(self):
        os.mkdir(f'{os.getcwd()}/metrics/random_forest_test')
        os.mkdir(f'{os.getcwd()}/metrics/random_forest_train')
        os.mkdir(f'{os.getcwd()}/metrics/random_forest_weights')
        print('metrics subdir already setup')

    def _check_graphs_dir(self):
        if 'graphs' not in self.dir_list:
            os.mkdir('graphs')
            print('graph dir setup succesfully')
            self._check_subgraphs_dir()
        else:
            print('graph dir already setup')

    def _check_subgraphs_dir(self):
        os.mkdir(f'{os.getcwd()}/graphs/clasification_heatmap')
        os.mkdir(f'{os.getcwd()}/graphs/forest_test')
        os.mkdir(f'{os.getcwd()}/graphs/kmeans')
        os.mkdir(f'{os.getcwd()}/graphs/normal_analysis')
        os.mkdir(f'{os.getcwd()}/graphs/pairplots')
        os.mkdir(f'{os.getcwd()}/graphs/tsne')
        print('graph subdirs setup succesfully')
